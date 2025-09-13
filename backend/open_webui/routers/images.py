import asyncio
import base64
import io
import json
import logging
import time
import mimetypes
import re
from pathlib import Path
import os
from typing import Optional

from urllib.parse import quote
import requests
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    UploadFile,
)

from open_webui.config import CACHE_DIR
from open_webui.constants import ERROR_MESSAGES
from open_webui.env import ENABLE_FORWARD_USER_INFO_HEADERS, SRC_LOG_LEVELS
from open_webui.routers.files import upload_file_handler
from open_webui.utils.auth import get_admin_user, get_verified_user
from open_webui.utils.images.comfyui import (
    ComfyUIGenerateImageForm,
    ComfyUIWorkflow,
    comfyui_generate_image,
)
from pydantic import BaseModel

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["IMAGES"])

IMAGE_CACHE_DIR = CACHE_DIR / "image" / "generations"
IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


router = APIRouter()


async def generate_images_grid(
    request: Request,
    form_data,
    user,
    progress_cb=None,
):
    """Helper to generate images via AIPowerGrid with optional progress callback.
    progress_cb: async callable(dict) — will be awaited with status dicts.
    """
    size = "512x512"
    if request.app.state.config.IMAGE_SIZE and "x" in request.app.state.config.IMAGE_SIZE:
        size = request.app.state.config.IMAGE_SIZE
    if getattr(form_data, "size", None) and "x" in form_data.size:
        size = form_data.size
    width, height = tuple(map(int, size.split("x")))

    last_http_response = None
    try:
        AIPG_API_KEY = os.getenv("AIPG_API_KEY", "")
        if not AIPG_API_KEY:
            raise HTTPException(status_code=401, detail="AIPG_API_KEY is not set in environment")

        GRID_BASE_URL = "https://api.aipowergrid.io"
        ENV_GEN = os.getenv("AIPG_IMAGE_GENERATE_ENDPOINT", "").strip()
        ENV_STATUS = os.getenv("AIPG_IMAGE_STATUS_ENDPOINT", "").strip()
        candidates = []
        if ENV_GEN and ENV_STATUS:
            candidates.append((ENV_GEN, ENV_STATUS))
        candidates += [
            (f"{GRID_BASE_URL}/api/v2/generate/async", f"{GRID_BASE_URL}/api/v2/generate/status"),
            (f"{GRID_BASE_URL}/v2/generate/async", f"{GRID_BASE_URL}/v2/generate/status"),
            (f"{GRID_BASE_URL}/api/v2/generate/image/async", f"{GRID_BASE_URL}/api/v2/generate/image/status"),
            (f"{GRID_BASE_URL}/api/v2/generate/images/async", f"{GRID_BASE_URL}/api/v2/generate/images/status"),
            (f"{GRID_BASE_URL}/api/v2/images/generate/async", f"{GRID_BASE_URL}/api/v2/images/generate/status"),
            (f"{GRID_BASE_URL}/api/v2/generate/img/async", f"{GRID_BASE_URL}/api/v2/generate/img/status"),
            (f"{GRID_BASE_URL}/v2/generate/image/async", f"{GRID_BASE_URL}/v2/generate/image/status"),
            (f"{GRID_BASE_URL}/v2/generate/images/async", f"{GRID_BASE_URL}/v2/generate/images/status"),
            (f"{GRID_BASE_URL}/v2/images/generate/async", f"{GRID_BASE_URL}/v2/images/generate/status"),
            (f"{GRID_BASE_URL}/v2/generate/img/async", f"{GRID_BASE_URL}/v2/generate/img/status"),
        ]

        headers = {"apikey": AIPG_API_KEY, "Content-Type": "application/json"}
        model_name = (form_data.model or "").replace("grid_", "") if getattr(form_data, "model", None) else None
        params = {"width": width, "height": height}
        if request.app.state.config.IMAGE_STEPS is not None:
            params["steps"] = request.app.state.config.IMAGE_STEPS
        data = {
            "prompt": form_data.prompt,
            **({"negative_prompt": form_data.negative_prompt} if getattr(form_data, "negative_prompt", None) else {}),
            "params": params,
            **({"models": [model_name]} if model_name else {}),
            "trusted_workers": False,
            "slow_workers": True,
        }

        # If no model specified, pick the first available image model from AIPG
        if not model_name:
            try:
                models_res = requests.get(
                    f"{GRID_BASE_URL}/api/v2/status/models?type=image&model_state=all",
                    headers={
                        "apikey": AIPG_API_KEY,
                        "accept": "application/json",
                        "Client-Agent": "openwebui:0:unknown",
                    },
                    timeout=30,
                )
                if models_res.status_code == 200:
                    arr = models_res.json() if isinstance(models_res.json(), list) else []
                    first = next((m for m in arr if m.get("name")), None)
                    if first and first.get("name"):
                        model_name = first["name"]
                        data["models"] = [model_name]
                        log.info(f"AIPG: no model specified; using first available '{model_name}'")
            except Exception:
                pass

        # Submit
        submit_res = None
        used_status_endpoint = None
        submit_error = None
        for GEN_URL, STATUS_URL in candidates:
            try:
                # Log the submit attempt (without API key)
                prompt_preview = (form_data.prompt[:120] + "...") if len(form_data.prompt) > 120 else form_data.prompt
                log.info(
                    f"AIPG submit -> url={GEN_URL} model={model_name or 'default'} size={width}x{height} steps={params.get('steps')} prompt='{prompt_preview}'"
                )
            except Exception:
                pass
            r = await asyncio.to_thread(
                requests.post,
                url=GEN_URL,
                json=data,
                headers=headers,
                timeout=60,
            )
            last_http_response = r
            try:
                body_preview = r.text[:500]
                log.info(f"AIPG submit <- status={r.status_code} body='{body_preview}'")
            except Exception:
                log.info(f"AIPG submit <- status={r.status_code} (no body preview)")
            if r.status_code in (200, 202):
                try:
                    submit_res = r.json()
                    used_status_endpoint = STATUS_URL
                    log.info(f"AIPG status base set -> {used_status_endpoint}")
                    break
                except Exception:
                    submit_error = r.text
                    continue
            elif r.status_code == 404:
                submit_error = r.text
                continue
            else:
                try:
                    submit_error = r.json()
                except Exception:
                    submit_error = r.text
                continue
        if not submit_res:
            raise HTTPException(status_code=500, detail=f"Grid image submit failed: {submit_error}")

        request_id = submit_res.get("id")
        if not request_id:
            raise HTTPException(status_code=500, detail="No Grid image request id")

        # Poll
        start_ts = time.time()
        max_attempts = 120
        attempt = 0
        images = []
        sleep_s = 2.0
        submit_wait_hint = submit_res.get("wait_time") or submit_res.get("wait") or submit_res.get("eta")
        if isinstance(submit_wait_hint, (int, float)):
            sleep_s = max(1.0, min(float(submit_wait_hint), 15.0))
        last_wait = None

        while attempt < max_attempts:
            # Progress callback
            if progress_cb and (last_wait is not None or submit_wait_hint is not None):
                try:
                    await progress_cb(
                        {
                            "type": "status",
                            "data": {
                                "description": f"In queue ~{int((last_wait or submit_wait_hint) or sleep_s)}s",
                                "done": False,
                                "wait_time": (last_wait or submit_wait_hint),
                            },
                        }
                    )
                except Exception:
                    pass

            await asyncio.sleep(sleep_s)
            attempt += 1
            status_url = f"{used_status_endpoint}/{request_id}"
            s = await asyncio.to_thread(
                requests.get,
                url=status_url,
                headers=headers,
                timeout=60,
            )
            last_http_response = s
            try:
                log.debug(f"AIPG poll -> {status_url}")
                log.debug(f"AIPG poll <- status={s.status_code} body='{s.text[:400]}'")
            except Exception:
                pass
            if s.status_code == 429:
                retry_after = 0
                try:
                    retry_after = int(s.headers.get("Retry-After", "0"))
                except Exception:
                    retry_after = 0
                sleep_s = max(sleep_s * 2.0, float(retry_after) if retry_after > 0 else 5.0)
                sleep_s = min(sleep_s, 20.0)
                continue
            if s.status_code == 404:
                raise HTTPException(status_code=500, detail="Grid image request not found")
            s.raise_for_status()
            status_data = s.json()
            wt = status_data.get("wait_time") or status_data.get("wait") or status_data.get("eta")
            if isinstance(wt, (int, float)):
                last_wait = float(wt)
                sleep_s = max(1.0, min(float(wt), 15.0))

            if status_data.get("done"):
                gens = status_data.get("generations", []) or status_data.get("images", [])
                log.info(f"AIPG done: generations={len(gens)}")
                for gen in gens:
                    img_data = None
                    content_type = None
                    if isinstance(gen, dict):
                        # Common fields across providers
                        if gen.get("url"):
                            img_data, content_type = load_url_image_data(gen["url"])  # may be public
                        elif gen.get("img"):
                            # AIPG returns 'img' for a direct URL in some responses
                            img_data, content_type = load_url_image_data(gen["img"])  # may be public
                        elif gen.get("image"):
                            img_data, content_type = load_b64_image_data(gen["image"])
                        elif gen.get("b64_json"):
                            img_data, content_type = load_b64_image_data(gen["b64_json"])
                        elif gen.get("data"):
                            img_data, content_type = load_b64_image_data(gen["data"])
                    elif isinstance(gen, str):
                        img_data, content_type = load_b64_image_data(gen)
                    if img_data and content_type:
                        url = upload_image(
                            request,
                            img_data,
                            content_type,
                            {
                                "provider": "grid",
                                "job_id": request_id,
                                "width": width,
                                "height": height,
                                **({"wait_time": last_wait} if last_wait is not None else {}),
                                **({"submit_wait": submit_wait_hint} if submit_wait_hint is not None else {}),
                            },
                            user,
                        )
                        images.append(
                            {
                                "url": url,
                                "meta": {
                                    "provider": "grid",
                                    "job_id": request_id,
                                    **({"wait_time": last_wait} if last_wait is not None else {}),
                                    **({"submit_wait": submit_wait_hint} if submit_wait_hint is not None else {}),
                                },
                            }
                        )
                if images:
                    log.info(f"AIPG saved images: {len(images)}")
                    return images
                # done but no decodable images; log keys to aid debugging
                try:
                    sample = gens[0] if gens else {}
                    log.error(f"AIPG done but no images decoded; sample keys={list(sample.keys()) if isinstance(sample, dict) else type(sample)}")
                except Exception:
                    pass
                break
            if status_data.get("faulted"):
                log.error(f"AIPG faulted: {status_data}")
                raise HTTPException(status_code=500, detail="Grid image generation failed")

        raise HTTPException(status_code=408, detail="Grid image generation timeout")
    except HTTPException:
        raise
    except Exception as e:
        detail_msg = str(e)
        if last_http_response is not None:
            try:
                data = last_http_response.json()
                if isinstance(data, dict):
                    if "error" in data:
                        err_val = data["error"]
                        if isinstance(err_val, dict):
                            detail_msg = err_val.get("message", detail_msg)
                        else:
                            detail_msg = str(err_val)
                else:
                    detail_msg = str(data)
            except Exception:
                try:
                    detail_msg = last_http_response.text
                except Exception:
                    pass
        raise HTTPException(status_code=400, detail=ERROR_MESSAGES.DEFAULT(detail_msg))


@router.get("/config")
async def get_config(request: Request, user=Depends(get_admin_user)):
    return {
        "enabled": request.app.state.config.ENABLE_IMAGE_GENERATION,
        "engine": request.app.state.config.IMAGE_GENERATION_ENGINE,
        "prompt_generation": request.app.state.config.ENABLE_IMAGE_PROMPT_GENERATION,
        "openai": {
            "OPENAI_API_BASE_URL": request.app.state.config.IMAGES_OPENAI_API_BASE_URL,
            "OPENAI_API_VERSION": request.app.state.config.IMAGES_OPENAI_API_VERSION,
            "OPENAI_API_KEY": request.app.state.config.IMAGES_OPENAI_API_KEY,
        },
        "automatic1111": {
            "AUTOMATIC1111_BASE_URL": request.app.state.config.AUTOMATIC1111_BASE_URL,
            "AUTOMATIC1111_API_AUTH": request.app.state.config.AUTOMATIC1111_API_AUTH,
            "AUTOMATIC1111_CFG_SCALE": request.app.state.config.AUTOMATIC1111_CFG_SCALE,
            "AUTOMATIC1111_SAMPLER": request.app.state.config.AUTOMATIC1111_SAMPLER,
            "AUTOMATIC1111_SCHEDULER": request.app.state.config.AUTOMATIC1111_SCHEDULER,
        },
        "comfyui": {
            "COMFYUI_BASE_URL": request.app.state.config.COMFYUI_BASE_URL,
            "COMFYUI_API_KEY": request.app.state.config.COMFYUI_API_KEY,
            "COMFYUI_WORKFLOW": request.app.state.config.COMFYUI_WORKFLOW,
            "COMFYUI_WORKFLOW_NODES": request.app.state.config.COMFYUI_WORKFLOW_NODES,
        },
        "gemini": {
            "GEMINI_API_BASE_URL": request.app.state.config.IMAGES_GEMINI_API_BASE_URL,
            "GEMINI_API_KEY": request.app.state.config.IMAGES_GEMINI_API_KEY,
        },
    }


class OpenAIConfigForm(BaseModel):
    OPENAI_API_BASE_URL: str
    OPENAI_API_VERSION: str
    OPENAI_API_KEY: str


class Automatic1111ConfigForm(BaseModel):
    AUTOMATIC1111_BASE_URL: str
    AUTOMATIC1111_API_AUTH: str
    AUTOMATIC1111_CFG_SCALE: Optional[str | float | int]
    AUTOMATIC1111_SAMPLER: Optional[str]
    AUTOMATIC1111_SCHEDULER: Optional[str]


class ComfyUIConfigForm(BaseModel):
    COMFYUI_BASE_URL: str
    COMFYUI_API_KEY: str
    COMFYUI_WORKFLOW: str
    COMFYUI_WORKFLOW_NODES: list[dict]


class GeminiConfigForm(BaseModel):
    GEMINI_API_BASE_URL: str
    GEMINI_API_KEY: str


class ConfigForm(BaseModel):
    enabled: bool
    engine: str
    prompt_generation: bool
    openai: OpenAIConfigForm
    automatic1111: Automatic1111ConfigForm
    comfyui: ComfyUIConfigForm
    gemini: GeminiConfigForm


@router.post("/config/update")
async def update_config(
    request: Request, form_data: ConfigForm, user=Depends(get_admin_user)
):
    request.app.state.config.IMAGE_GENERATION_ENGINE = form_data.engine
    request.app.state.config.ENABLE_IMAGE_GENERATION = form_data.enabled

    request.app.state.config.ENABLE_IMAGE_PROMPT_GENERATION = (
        form_data.prompt_generation
    )

    request.app.state.config.IMAGES_OPENAI_API_BASE_URL = (
        form_data.openai.OPENAI_API_BASE_URL
    )
    request.app.state.config.IMAGES_OPENAI_API_VERSION = (
        form_data.openai.OPENAI_API_VERSION
    )
    request.app.state.config.IMAGES_OPENAI_API_KEY = form_data.openai.OPENAI_API_KEY

    request.app.state.config.IMAGES_GEMINI_API_BASE_URL = (
        form_data.gemini.GEMINI_API_BASE_URL
    )
    request.app.state.config.IMAGES_GEMINI_API_KEY = form_data.gemini.GEMINI_API_KEY

    request.app.state.config.AUTOMATIC1111_BASE_URL = (
        form_data.automatic1111.AUTOMATIC1111_BASE_URL
    )
    request.app.state.config.AUTOMATIC1111_API_AUTH = (
        form_data.automatic1111.AUTOMATIC1111_API_AUTH
    )

    request.app.state.config.AUTOMATIC1111_CFG_SCALE = (
        float(form_data.automatic1111.AUTOMATIC1111_CFG_SCALE)
        if form_data.automatic1111.AUTOMATIC1111_CFG_SCALE
        else None
    )
    request.app.state.config.AUTOMATIC1111_SAMPLER = (
        form_data.automatic1111.AUTOMATIC1111_SAMPLER
        if form_data.automatic1111.AUTOMATIC1111_SAMPLER
        else None
    )
    request.app.state.config.AUTOMATIC1111_SCHEDULER = (
        form_data.automatic1111.AUTOMATIC1111_SCHEDULER
        if form_data.automatic1111.AUTOMATIC1111_SCHEDULER
        else None
    )

    request.app.state.config.COMFYUI_BASE_URL = (
        form_data.comfyui.COMFYUI_BASE_URL.strip("/")
    )
    request.app.state.config.COMFYUI_API_KEY = form_data.comfyui.COMFYUI_API_KEY

    request.app.state.config.COMFYUI_WORKFLOW = form_data.comfyui.COMFYUI_WORKFLOW
    request.app.state.config.COMFYUI_WORKFLOW_NODES = (
        form_data.comfyui.COMFYUI_WORKFLOW_NODES
    )

    return {
        "enabled": request.app.state.config.ENABLE_IMAGE_GENERATION,
        "engine": request.app.state.config.IMAGE_GENERATION_ENGINE,
        "prompt_generation": request.app.state.config.ENABLE_IMAGE_PROMPT_GENERATION,
        "openai": {
            "OPENAI_API_BASE_URL": request.app.state.config.IMAGES_OPENAI_API_BASE_URL,
            "OPENAI_API_VERSION": request.app.state.config.IMAGES_OPENAI_API_VERSION,
            "OPENAI_API_KEY": request.app.state.config.IMAGES_OPENAI_API_KEY,
        },
        "automatic1111": {
            "AUTOMATIC1111_BASE_URL": request.app.state.config.AUTOMATIC1111_BASE_URL,
            "AUTOMATIC1111_API_AUTH": request.app.state.config.AUTOMATIC1111_API_AUTH,
            "AUTOMATIC1111_CFG_SCALE": request.app.state.config.AUTOMATIC1111_CFG_SCALE,
            "AUTOMATIC1111_SAMPLER": request.app.state.config.AUTOMATIC1111_SAMPLER,
            "AUTOMATIC1111_SCHEDULER": request.app.state.config.AUTOMATIC1111_SCHEDULER,
        },
        "comfyui": {
            "COMFYUI_BASE_URL": request.app.state.config.COMFYUI_BASE_URL,
            "COMFYUI_API_KEY": request.app.state.config.COMFYUI_API_KEY,
            "COMFYUI_WORKFLOW": request.app.state.config.COMFYUI_WORKFLOW,
            "COMFYUI_WORKFLOW_NODES": request.app.state.config.COMFYUI_WORKFLOW_NODES,
        },
        "gemini": {
            "GEMINI_API_BASE_URL": request.app.state.config.IMAGES_GEMINI_API_BASE_URL,
            "GEMINI_API_KEY": request.app.state.config.IMAGES_GEMINI_API_KEY,
        },
    }


def get_automatic1111_api_auth(request: Request):
    if request.app.state.config.AUTOMATIC1111_API_AUTH is None:
        return ""
    else:
        auth1111_byte_string = request.app.state.config.AUTOMATIC1111_API_AUTH.encode(
            "utf-8"
        )
        auth1111_base64_encoded_bytes = base64.b64encode(auth1111_byte_string)
        auth1111_base64_encoded_string = auth1111_base64_encoded_bytes.decode("utf-8")
        return f"Basic {auth1111_base64_encoded_string}"


@router.get("/config/url/verify")
async def verify_url(request: Request, user=Depends(get_admin_user)):
    if request.app.state.config.IMAGE_GENERATION_ENGINE == "automatic1111":
        try:
            r = requests.get(
                url=f"{request.app.state.config.AUTOMATIC1111_BASE_URL}/sdapi/v1/options",
                headers={"authorization": get_automatic1111_api_auth(request)},
            )
            r.raise_for_status()
            return True
        except Exception:
            request.app.state.config.ENABLE_IMAGE_GENERATION = False
            raise HTTPException(status_code=400, detail=ERROR_MESSAGES.INVALID_URL)
    elif request.app.state.config.IMAGE_GENERATION_ENGINE == "comfyui":

        headers = None
        if request.app.state.config.COMFYUI_API_KEY:
            headers = {
                "Authorization": f"Bearer {request.app.state.config.COMFYUI_API_KEY}"
            }

        try:
            r = requests.get(
                url=f"{request.app.state.config.COMFYUI_BASE_URL}/object_info",
                headers=headers,
            )
            r.raise_for_status()
            return True
        except Exception:
            request.app.state.config.ENABLE_IMAGE_GENERATION = False
            raise HTTPException(status_code=400, detail=ERROR_MESSAGES.INVALID_URL)
    else:
        return True


def set_image_model(request: Request, model: str):
    log.info(f"Setting image model to {model}")
    request.app.state.config.IMAGE_GENERATION_MODEL = model
    if request.app.state.config.IMAGE_GENERATION_ENGINE in ["", "automatic1111"]:
        api_auth = get_automatic1111_api_auth(request)
        r = requests.get(
            url=f"{request.app.state.config.AUTOMATIC1111_BASE_URL}/sdapi/v1/options",
            headers={"authorization": api_auth},
        )
        options = r.json()
        if model != options["sd_model_checkpoint"]:
            options["sd_model_checkpoint"] = model
            r = requests.post(
                url=f"{request.app.state.config.AUTOMATIC1111_BASE_URL}/sdapi/v1/options",
                json=options,
                headers={"authorization": api_auth},
            )
    return request.app.state.config.IMAGE_GENERATION_MODEL


def get_image_model(request):
    if request.app.state.config.IMAGE_GENERATION_ENGINE == "openai":
        return (
            request.app.state.config.IMAGE_GENERATION_MODEL
            if request.app.state.config.IMAGE_GENERATION_MODEL
            else "dall-e-2"
        )
    elif request.app.state.config.IMAGE_GENERATION_ENGINE == "gemini":
        return (
            request.app.state.config.IMAGE_GENERATION_MODEL
            if request.app.state.config.IMAGE_GENERATION_MODEL
            else "imagen-3.0-generate-002"
        )
    elif request.app.state.config.IMAGE_GENERATION_ENGINE == "comfyui":
        return (
            request.app.state.config.IMAGE_GENERATION_MODEL
            if request.app.state.config.IMAGE_GENERATION_MODEL
            else ""
        )
    elif (
        request.app.state.config.IMAGE_GENERATION_ENGINE == "automatic1111"
        or request.app.state.config.IMAGE_GENERATION_ENGINE == ""
    ):
        try:
            r = requests.get(
                url=f"{request.app.state.config.AUTOMATIC1111_BASE_URL}/sdapi/v1/options",
                headers={"authorization": get_automatic1111_api_auth(request)},
            )
            options = r.json()
            return options["sd_model_checkpoint"]
        except Exception as e:
            request.app.state.config.ENABLE_IMAGE_GENERATION = False
            raise HTTPException(status_code=400, detail=ERROR_MESSAGES.DEFAULT(e))


class ImageConfigForm(BaseModel):
    MODEL: str
    IMAGE_SIZE: str
    IMAGE_STEPS: int


@router.get("/image/config")
async def get_image_config(request: Request, user=Depends(get_admin_user)):
    return {
        "MODEL": request.app.state.config.IMAGE_GENERATION_MODEL,
        "IMAGE_SIZE": request.app.state.config.IMAGE_SIZE,
        "IMAGE_STEPS": request.app.state.config.IMAGE_STEPS,
    }


@router.post("/image/config/update")
async def update_image_config(
    request: Request, form_data: ImageConfigForm, user=Depends(get_admin_user)
):
    set_image_model(request, form_data.MODEL)

    if form_data.IMAGE_SIZE == "auto" and form_data.MODEL != "gpt-image-1":
        raise HTTPException(
            status_code=400,
            detail=ERROR_MESSAGES.INCORRECT_FORMAT(
                "  (auto is only allowed with gpt-image-1)."
            ),
        )

    pattern = r"^\d+x\d+$"
    if form_data.IMAGE_SIZE == "auto" or re.match(pattern, form_data.IMAGE_SIZE):
        request.app.state.config.IMAGE_SIZE = form_data.IMAGE_SIZE
    else:
        raise HTTPException(
            status_code=400,
            detail=ERROR_MESSAGES.INCORRECT_FORMAT("  (e.g., 512x512)."),
        )

    if form_data.IMAGE_STEPS >= 0:
        request.app.state.config.IMAGE_STEPS = form_data.IMAGE_STEPS
    else:
        raise HTTPException(
            status_code=400,
            detail=ERROR_MESSAGES.INCORRECT_FORMAT("  (e.g., 50)."),
        )

    return {
        "MODEL": request.app.state.config.IMAGE_GENERATION_MODEL,
        "IMAGE_SIZE": request.app.state.config.IMAGE_SIZE,
        "IMAGE_STEPS": request.app.state.config.IMAGE_STEPS,
    }


@router.get("/models")
def get_models(request: Request, user=Depends(get_verified_user)):
    try:
        if request.app.state.config.IMAGE_GENERATION_ENGINE == "openai":
            return [
                {"id": "dall-e-2", "name": "DALL·E 2"},
                {"id": "dall-e-3", "name": "DALL·E 3"},
                {"id": "gpt-image-1", "name": "GPT-IMAGE 1"},
            ]
        elif request.app.state.config.IMAGE_GENERATION_ENGINE == "gemini":
            return [
                {"id": "imagen-3.0-generate-002", "name": "imagen-3.0 generate-002"},
            ]
        elif request.app.state.config.IMAGE_GENERATION_ENGINE == "comfyui":
            # TODO - get models from comfyui
            headers = {
                "Authorization": f"Bearer {request.app.state.config.COMFYUI_API_KEY}"
            }
            r = requests.get(
                url=f"{request.app.state.config.COMFYUI_BASE_URL}/object_info",
                headers=headers,
            )
            info = r.json()

            workflow = json.loads(request.app.state.config.COMFYUI_WORKFLOW)
            model_node_id = None

            for node in request.app.state.config.COMFYUI_WORKFLOW_NODES:
                if node["type"] == "model":
                    if node["node_ids"]:
                        model_node_id = node["node_ids"][0]
                    break

            if model_node_id:
                model_list_key = None

                log.info(workflow[model_node_id]["class_type"])
                for key in info[workflow[model_node_id]["class_type"]]["input"][
                    "required"
                ]:
                    if "_name" in key:
                        model_list_key = key
                        break

                if model_list_key:
                    return list(
                        map(
                            lambda model: {"id": model, "name": model},
                            info[workflow[model_node_id]["class_type"]]["input"][
                                "required"
                            ][model_list_key][0],
                        )
                    )
            else:
                return list(
                    map(
                        lambda model: {"id": model, "name": model},
                        info["CheckpointLoaderSimple"]["input"]["required"][
                            "ckpt_name"
                        ][0],
                    )
                )
        elif request.app.state.config.IMAGE_GENERATION_ENGINE == "grid":
            # Fetch image models from AIPowerGrid (best-effort)
            try:
                AIPG_API_KEY = os.getenv("AIPG_API_KEY", "")
                if not AIPG_API_KEY:
                    raise Exception("AIPG_API_KEY not set")

                GRID_BASE_URL = "https://api.aipowergrid.io"
                # Prefer canonical endpoint with model_state=all
                model_endpoints = [
                    f"{GRID_BASE_URL}/api/v2/status/models?type=image&model_state=all",
                    f"{GRID_BASE_URL}/v2/status/models?type=image&model_state=all",
                    f"{GRID_BASE_URL}/api/v2/status/models?type=image",
                    f"{GRID_BASE_URL}/v2/status/models?type=image",
                ]
                data = []
                for me in model_endpoints:
                    try:
                        r = requests.get(
                            me,
                            headers={
                                "apikey": AIPG_API_KEY,
                                "accept": "application/json",
                                "Client-Agent": "openwebui:0:unknown",
                            },
                            timeout=30,
                        )
                        if r.status_code == 200:
                            jr = r.json()
                            if isinstance(jr, list):
                                data = jr
                                break
                    except Exception:
                        continue
                models = []
                for m in data:
                    # Expect keys like name, type='image'
                    name = m.get("name") or m.get("model") or None
                    if not name:
                        continue
                    models.append({"id": f"grid_{name}", "name": name})
                if models:
                    return models
                # Fallback if API returned empty
                return [{"id": "grid_default", "name": "Grid Default"}]
            except Exception as e:
                log.debug(f"AIPG image models fetch failed: {e}")
                return [{"id": "grid_default", "name": "Grid Default"}]
        elif (
            request.app.state.config.IMAGE_GENERATION_ENGINE == "automatic1111"
            or request.app.state.config.IMAGE_GENERATION_ENGINE == ""
        ):
            r = requests.get(
                url=f"{request.app.state.config.AUTOMATIC1111_BASE_URL}/sdapi/v1/sd-models",
                headers={"authorization": get_automatic1111_api_auth(request)},
            )
            models = r.json()
            return list(
                map(
                    lambda model: {"id": model["title"], "name": model["model_name"]},
                    models,
                )
            )
    except Exception as e:
        request.app.state.config.ENABLE_IMAGE_GENERATION = False
        raise HTTPException(status_code=400, detail=ERROR_MESSAGES.DEFAULT(e))


class GenerateImageForm(BaseModel):
    model: Optional[str] = None
    prompt: str
    size: Optional[str] = None
    n: int = 1
    negative_prompt: Optional[str] = None


def load_b64_image_data(b64_str):
    try:
        if "," in b64_str:
            header, encoded = b64_str.split(",", 1)
            mime_type = header.split(";")[0].lstrip("data:")
            img_data = base64.b64decode(encoded)
        else:
            mime_type = "image/png"
            img_data = base64.b64decode(b64_str)
        return img_data, mime_type
    except Exception as e:
        log.exception(f"Error loading image data: {e}")
        return None, None


def load_url_image_data(url, headers=None):
    try:
        if headers:
            r = requests.get(url, headers=headers)
        else:
            r = requests.get(url)

        r.raise_for_status()
        if r.headers["content-type"].split("/")[0] == "image":
            mime_type = r.headers["content-type"]
            return r.content, mime_type
        else:
            log.error("Url does not point to an image.")
            return None

    except Exception as e:
        log.exception(f"Error saving image: {e}")
        return None


def upload_image(request, image_data, content_type, metadata, user):
    image_format = mimetypes.guess_extension(content_type)
    file = UploadFile(
        file=io.BytesIO(image_data),
        filename=f"generated-image{image_format}",  # will be converted to a unique ID on upload_file
        headers={
            "content-type": content_type,
        },
    )
    file_item = upload_file_handler(
        request,
        file=file,
        metadata=metadata,
        process=False,
        user=user,
    )
    url = request.app.url_path_for("get_file_content_by_id", id=file_item.id)
    return url


@router.post("/generations")
async def image_generations(
    request: Request,
    form_data: GenerateImageForm,
    user=Depends(get_verified_user),
):
    # if IMAGE_SIZE = 'auto', default WidthxHeight to the 512x512 default
    # This is only relevant when the user has set IMAGE_SIZE to 'auto' with an
    # image model other than gpt-image-1, which is warned about on settings save

    size = "512x512"
    if (
        request.app.state.config.IMAGE_SIZE
        and "x" in request.app.state.config.IMAGE_SIZE
    ):
        size = request.app.state.config.IMAGE_SIZE

    if form_data.size and "x" in form_data.size:
        size = form_data.size

    width, height = tuple(map(int, size.split("x")))

    r = None
    last_http_response = None
    try:
        if request.app.state.config.IMAGE_GENERATION_ENGINE == "openai":
            headers = {}
            headers["Authorization"] = (
                f"Bearer {request.app.state.config.IMAGES_OPENAI_API_KEY}"
            )
            headers["Content-Type"] = "application/json"

            if ENABLE_FORWARD_USER_INFO_HEADERS:
                headers["X-OpenWebUI-User-Name"] = quote(user.name, safe=" ")
                headers["X-OpenWebUI-User-Id"] = user.id
                headers["X-OpenWebUI-User-Email"] = user.email
                headers["X-OpenWebUI-User-Role"] = user.role

            data = {
                "model": (
                    request.app.state.config.IMAGE_GENERATION_MODEL
                    if request.app.state.config.IMAGE_GENERATION_MODEL != ""
                    else "dall-e-2"
                ),
                "prompt": form_data.prompt,
                "n": form_data.n,
                "size": (
                    form_data.size
                    if form_data.size
                    else request.app.state.config.IMAGE_SIZE
                ),
                **(
                    {}
                    if "gpt-image-1" in request.app.state.config.IMAGE_GENERATION_MODEL
                    else {"response_format": "b64_json"}
                ),
            }

            api_version_query_param = ""
            if request.app.state.config.IMAGES_OPENAI_API_VERSION:
                api_version_query_param = (
                    f"?api-version={request.app.state.config.IMAGES_OPENAI_API_VERSION}"
                )

            # Use asyncio.to_thread for the requests.post call
            r = await asyncio.to_thread(
                requests.post,
                url=f"{request.app.state.config.IMAGES_OPENAI_API_BASE_URL}/images/generations{api_version_query_param}",
                json=data,
                headers=headers,
            )
            last_http_response = r

            r.raise_for_status()
            res = r.json()

            images = []

            for image in res["data"]:
                if image_url := image.get("url", None):
                    image_data, content_type = load_url_image_data(image_url, headers)
                else:
                    image_data, content_type = load_b64_image_data(image["b64_json"])

                url = upload_image(request, image_data, content_type, data, user)
                images.append({"url": url})
            return images

        elif request.app.state.config.IMAGE_GENERATION_ENGINE == "gemini":
            headers = {}
            headers["Content-Type"] = "application/json"
            headers["x-goog-api-key"] = request.app.state.config.IMAGES_GEMINI_API_KEY

            model = get_image_model(request)
            data = {
                "instances": {"prompt": form_data.prompt},
                "parameters": {
                    "sampleCount": form_data.n,
                    "outputOptions": {"mimeType": "image/png"},
                },
            }

            # Use asyncio.to_thread for the requests.post call
            r = await asyncio.to_thread(
                requests.post,
                url=f"{request.app.state.config.IMAGES_GEMINI_API_BASE_URL}/models/{model}:predict",
                json=data,
                headers=headers,
            )
            last_http_response = r

            r.raise_for_status()
            res = r.json()

            images = []
            for image in res["predictions"]:
                image_data, content_type = load_b64_image_data(
                    image["bytesBase64Encoded"]
                )
                url = upload_image(request, image_data, content_type, data, user)
                images.append({"url": url})

            return images

        elif request.app.state.config.IMAGE_GENERATION_ENGINE == "comfyui":
            data = {
                "prompt": form_data.prompt,
                "width": width,
                "height": height,
                "n": form_data.n,
            }

            if request.app.state.config.IMAGE_STEPS is not None:
                data["steps"] = request.app.state.config.IMAGE_STEPS

            if form_data.negative_prompt is not None:
                data["negative_prompt"] = form_data.negative_prompt

            form_data = ComfyUIGenerateImageForm(
                **{
                    "workflow": ComfyUIWorkflow(
                        **{
                            "workflow": request.app.state.config.COMFYUI_WORKFLOW,
                            "nodes": request.app.state.config.COMFYUI_WORKFLOW_NODES,
                        }
                    ),
                    **data,
                }
            )
            res = await comfyui_generate_image(
                request.app.state.config.IMAGE_GENERATION_MODEL,
                form_data,
                user.id,
                request.app.state.config.COMFYUI_BASE_URL,
                request.app.state.config.COMFYUI_API_KEY,
            )
            log.debug(f"res: {res}")

            images = []

            for image in res["data"]:
                headers = None
                if request.app.state.config.COMFYUI_API_KEY:
                    headers = {
                        "Authorization": f"Bearer {request.app.state.config.COMFYUI_API_KEY}"
                    }

                image_data, content_type = load_url_image_data(image["url"], headers)
                url = upload_image(
                    request,
                    image_data,
                    content_type,
                    form_data.model_dump(exclude_none=True),
                    user,
                )
                images.append({"url": url})
            return images
        elif (
            request.app.state.config.IMAGE_GENERATION_ENGINE == "automatic1111"
            or request.app.state.config.IMAGE_GENERATION_ENGINE == ""
        ):
            if form_data.model:
                set_image_model(request, form_data.model)

            data = {
                "prompt": form_data.prompt,
                "batch_size": form_data.n,
                "width": width,
                "height": height,
            }

            if request.app.state.config.IMAGE_STEPS is not None:
                data["steps"] = request.app.state.config.IMAGE_STEPS

            if form_data.negative_prompt is not None:
                data["negative_prompt"] = form_data.negative_prompt

            if request.app.state.config.AUTOMATIC1111_CFG_SCALE:
                data["cfg_scale"] = request.app.state.config.AUTOMATIC1111_CFG_SCALE

            if request.app.state.config.AUTOMATIC1111_SAMPLER:
                data["sampler_name"] = request.app.state.config.AUTOMATIC1111_SAMPLER

            if request.app.state.config.AUTOMATIC1111_SCHEDULER:
                data["scheduler"] = request.app.state.config.AUTOMATIC1111_SCHEDULER

            # Use asyncio.to_thread for the requests.post call
            r = await asyncio.to_thread(
                requests.post,
                url=f"{request.app.state.config.AUTOMATIC1111_BASE_URL}/sdapi/v1/txt2img",
                json=data,
                headers={"authorization": get_automatic1111_api_auth(request)},
            )
            last_http_response = r

            res = r.json()
            log.debug(f"res: {res}")

            images = []

            for image in res["images"]:
                image_data, content_type = load_b64_image_data(image)
                url = upload_image(
                    request,
                    image_data,
                    content_type,
                    {**data, "info": res["info"]},
                    user,
                )
                images.append({"url": url})
            return images
        elif request.app.state.config.IMAGE_GENERATION_ENGINE == "grid":
            # Delegate to helper to allow progress callbacks and reuse logic
            return await generate_images_grid(request, form_data, user)
    except Exception as e:
        detail_msg = str(e)
        if last_http_response is not None:
            try:
                data = last_http_response.json()
                if isinstance(data, dict):
                    if "error" in data:
                        err_val = data["error"]
                        if isinstance(err_val, dict):
                            detail_msg = err_val.get("message", detail_msg)
                        else:
                            detail_msg = str(err_val)
                else:
                    detail_msg = str(data)
            except Exception:
                try:
                    detail_msg = last_http_response.text
                except Exception:
                    pass
        raise HTTPException(status_code=400, detail=ERROR_MESSAGES.DEFAULT(detail_msg))
