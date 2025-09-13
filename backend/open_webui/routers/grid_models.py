"""
Grid Models Integration for Open WebUI
Provides dynamic model loading from AI Power Grid API
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import List, Dict, Optional, Any
from urllib.parse import quote
import os

import aiohttp
from fastapi import APIRouter, HTTPException, Depends, Request

from pydantic import BaseModel

from open_webui.models.chats import ChatModel
from open_webui.models.models import Model
from open_webui.utils.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()

# Grid API configuration
# NOTE: API key is read from environment to avoid hardcoding secrets.
AIPG_API_KEY = os.getenv("AIPG_API_KEY", "")
GRID_BASE_URL = "https://api.aipowergrid.io"
GRID_MODELS_ENDPOINT = f"{GRID_BASE_URL}/api/v2/status/models?type=text"
GRID_WORKERS_ENDPOINT = f"{GRID_BASE_URL}/api/v2/workers?type=text"
GRID_GENERATE_ENDPOINT = f"{GRID_BASE_URL}/api/v2/generate/text/async"
GRID_STATUS_ENDPOINT = f"{GRID_BASE_URL}/api/v2/generate/text/status"

class GridModel(BaseModel):
    name: str
    count: int
    queued: float
    eta: int
    performance: float
    jobs: float = 0.0
    type: str = "text"

class GridWorker(BaseModel):
    name: str
    models: List[str]
    performance: str
    trusted: bool
    maintenance_mode: bool
    online: bool

class GridGenerationRequest(BaseModel):
    prompt: str
    params: Dict[str, Any]
    models: List[str]
    trusted_workers: bool = False
    slow_workers: bool = True
    workers: List[str] = []
    worker_blacklist: bool = False
    blacklist: List[str] = []

class GridModelsProvider:
    """Provider for dynamic Grid models"""
    
    def __init__(self):
        # Only include apikey header when available
        self.headers = {"apikey": AIPG_API_KEY} if AIPG_API_KEY else {}
        self.cached_models = []
        self.cache_time = 0
        self.cache_duration = 30  # 30 seconds cache
        
    async def get_models(self) -> List[GridModel]:
        """Get available models from Grid API with caching"""
        current_time = time.time()
        
        # Return cached models if still valid
        if self.cached_models and (current_time - self.cache_time) < self.cache_duration:
            return self.cached_models
            
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(GRID_MODELS_ENDPOINT, headers=self.headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = []
                        
                        for model_data in data:
                            if model_data.get("type") == "text":
                                model = GridModel(
                                    name=model_data.get("name", ""),
                                    count=model_data.get("count", 0),
                                    queued=model_data.get("queued", 0.0),
                                    eta=model_data.get("eta", 0),
                                    performance=model_data.get("performance", 0.0)
                                )
                                models.append(model)
                        
                        self.cached_models = models
                        self.cache_time = current_time
                        return models
                    else:
                        logger.error(f"Failed to fetch Grid models: {response.status}")
                        return self.cached_models  # Return cached if available
                        
        except Exception as e:
            logger.error(f"Error fetching Grid models: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return self.cached_models  # Return cached if available
    
    async def get_workers(self) -> List[GridWorker]:
        """Get available workers from Grid API"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(GRID_WORKERS_ENDPOINT, headers=self.headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        workers = []
                        
                        for worker_data in data:
                            if worker_data.get("type") == "text" and worker_data.get("online"):
                                worker = GridWorker(
                                    name=worker_data.get("name", ""),
                                    models=worker_data.get("models", []),
                                    performance=worker_data.get("performance", "0"),
                                    trusted=worker_data.get("trusted", False),
                                    maintenance_mode=worker_data.get("maintenance_mode", False),
                                    online=worker_data.get("online", False)
                                )
                                workers.append(worker)
                        
                        return workers
                    else:
                        logger.error(f"Failed to fetch Grid workers: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error fetching Grid workers: {e}")
            return []

# Global provider instance
grid_provider = GridModelsProvider()


def _extract_worker_from_status(status: dict) -> Optional[str]:
    """Best-effort extraction of worker identity from Grid status payload."""
    if not isinstance(status, dict):
        return None
    # Common possible keys seen in job status payloads
    for key in [
        "worker",
        "worker_name",
        "node",
        "hostname",
        "producer",
        "executor",
    ]:
        val = status.get(key)
        if isinstance(val, str) and val.strip():
            return val
    # Sometimes embedded in first generation payload
    gens = status.get("generations") or []
    if gens and isinstance(gens, list):
        gen0 = gens[0]
        for key in ["worker", "worker_name", "node", "hostname"]:
            val = gen0.get(key) if isinstance(gen0, dict) else None
            if isinstance(val, str) and val.strip():
                return val
    return None

@router.get("/api/v1/models/grid")
async def get_grid_models():
    """Get available Grid models for the dropdown"""
    try:
        if not grid_provider.headers.get("apikey"):
            raise HTTPException(status_code=401, detail="AIPG_API_KEY is not set in environment")
        models = await grid_provider.get_models()
        
        # Convert to Open WebUI model format
        openwebui_models = []
        for model in models:
            openwebui_model = {
                "id": f"grid_{model.name}",
                "name": model.name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "grid",
                "permission": [],
                "root": f"grid_{model.name}",
                "parent": None,
                "meta": {
                    "description": f"Grid model: {model.name}",
                    "queue_length": model.queued,
                    "eta": model.eta,
                    "performance": str(model.performance),
                    "worker_count": model.count
                }
            }
            openwebui_models.append(openwebui_model)
        
        return {
            "object": "list",
            "data": openwebui_models
        }
        
    except Exception as e:
        logger.error(f"Error getting Grid models: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch Grid models")

@router.get("/api/v1/models/grid/refresh")
async def refresh_grid_models():
    """Force refresh of Grid models cache"""
    try:
        if not grid_provider.headers.get("apikey"):
            raise HTTPException(status_code=401, detail="AIPG_API_KEY is not set in environment")
        # Clear cache to force refresh
        grid_provider.cached_models = []
        grid_provider.cache_time = 0
        
        models = await grid_provider.get_models()
        return {"message": f"Refreshed {len(models)} Grid models", "count": len(models)}
        
    except Exception as e:
        logger.error(f"Error refreshing Grid models: {e}")
        raise HTTPException(status_code=500, detail="Failed to refresh Grid models")

@router.get("/api/v1/models/grid/workers")
async def get_grid_workers():
    """Get available Grid workers"""
    try:
        if not grid_provider.headers.get("apikey"):
            raise HTTPException(status_code=401, detail="AIPG_API_KEY is not set in environment")
        workers = await grid_provider.get_workers()
        return {"workers": [worker.dict() for worker in workers]}
        
    except Exception as e:
        logger.error(f"Error getting Grid workers: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch Grid workers")

@router.post("/api/v1/chat/completions/grid")
async def grid_chat_completion(request: Request):
    """Handle Grid model chat completions"""
    try:
        if not grid_provider.headers.get("apikey"):
            raise HTTPException(status_code=401, detail="AIPG_API_KEY is not set in environment")
        body = await request.json()
        
        # Extract parameters
        model_name = body.get("model", "")
        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens", 1024)  # Set to 1024 for optimal performance
        temperature = body.get("temperature", 0.7)
        top_p = body.get("top_p", 0.9)
        
        # Get the last user message
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break
        
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")
        
        # Prepare Grid generation request
        grid_request = GridGenerationRequest(
            prompt=user_message,
            params={
                "max_length": min(max_tokens, 1024),  # Cap at 1024 for optimal performance
                "temperature": temperature,
                "top_p": top_p,
                "top_k": 40,
                "rep_pen": 1.1,
                "rep_pen_range": 256,
                "rep_pen_slope": 0.7,
                "tfs": 1,
                "top_a": 0,
                "typical": 1,
                "singleline": False,
                "frmttriminc": True,
                "frmtrmblln": False,
                "stop_sequence": ["You:", "Human:", "Assistant:", "###"]
            },
            models=[model_name.replace("grid_", "")],  # Remove grid_ prefix
            trusted_workers=False,
            slow_workers=True
        )
        
        # Submit generation request
        t_start = time.time()
        async with aiohttp.ClientSession() as session:
            async with session.post(
                GRID_GENERATE_ENDPOINT, 
                headers=grid_provider.headers, 
                json=grid_request.dict()
            ) as response:
                if response.status != 202:
                    raise HTTPException(status_code=500, detail="Failed to submit Grid generation")
                
                result = await response.json()
                request_id = result.get("id")
                
                if not request_id:
                    raise HTTPException(status_code=500, detail="No request ID received")
                
                # Poll for completion
                max_attempts = 60  # 5 minutes max
                attempts = 0
                
                while attempts < max_attempts:
                    await asyncio.sleep(5)  # Wait 5 seconds
                    attempts += 1
                    
                    status_url = f"{GRID_STATUS_ENDPOINT}/{request_id}"
                    async with session.get(status_url, headers=grid_provider.headers) as status_response:
                        if status_response.status == 200:
                            status_data = await status_response.json()
                            
                            if status_data.get("done"):
                                generations = status_data.get("generations", [])
                                if generations:
                                    generated_text = generations[0].get("text", "")
                                    t_end = time.time()
                                    elapsed_ms = int((t_end - t_start) * 1000)
                                    prompt_tokens = len(user_message.split())
                                    completion_tokens = len(generated_text.split())
                                    total_tokens = prompt_tokens + completion_tokens
                                    worker_name = _extract_worker_from_status(status_data)
                                    tps = (completion_tokens / max((t_end - t_start), 1e-6)) if completion_tokens > 0 else 0.0
                                    
                                    # Return in OpenAI-compatible format
                                    return {
                                        "id": f"grid_{request_id}",
                                        "object": "chat.completion",
                                        "created": int(time.time()),
                                        "model": model_name,
                                        "choices": [{
                                            "index": 0,
                                            "message": {
                                                "role": "assistant",
                                                "content": generated_text
                                            },
                                            "finish_reason": "stop"
                                        }],
                                        "usage": {
                                            "provider": "grid",
                                            "job_id": request_id,
                                            **({"worker": worker_name} if worker_name else {}),
                                            "elapsed_ms": elapsed_ms,
                                            "tokens_per_sec": tps,
                                            "prompt_tokens": prompt_tokens,
                                            "completion_tokens": completion_tokens,
                                            "total_tokens": total_tokens,
                                        }
                                    }
                                
                                # If no generations, check for errors
                                if status_data.get("faulted"):
                                    raise HTTPException(status_code=500, detail="Grid generation failed")
                                
                        elif status_response.status == 404:
                            raise HTTPException(status_code=500, detail="Grid request not found")
                
                # Timeout
                raise HTTPException(status_code=408, detail="Grid generation timeout")
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in Grid chat completion: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

def convert_messages_to_prompt(messages: List[dict]) -> str:
    """Convert conversation messages to a prompt format for Grid API, filtering out thinking tags"""
    if not messages:
        logger.warning("Empty messages list provided to convert_messages_to_prompt")
        return ""
    
    conversation_prompt = ""
    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        # Skip empty messages
        if not content or not content.strip():
            logger.warning(f"Empty content in message {i} with role '{role}'")
            continue
        
        # Filter out thinking tags to avoid context window bloat
        # Remove <think>...</think> and <details type="reasoning">...</details> tags
        import re
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        content = re.sub(r'<details[^>]*type="reasoning"[^>]*>.*?</details>', '', content, flags=re.DOTALL)
        content = re.sub(r'<details[^>]*>.*?<summary>Thinking</summary>.*?</details>', '', content, flags=re.DOTALL)
        
        # Skip if content is empty after filtering
        if not content.strip():
            continue
            
        if role == "user":
            conversation_prompt += f"Human: {content}\n"
        elif role == "assistant":
            conversation_prompt += f"Assistant: {content}\n"
        elif role == "system":
            conversation_prompt += f"System: {content}\n"
        else:
            logger.warning(f"Unknown role '{role}' in message {i}")
    
    logger.debug(f"Converted {len(messages)} messages to prompt: {conversation_prompt[:100]}...")
    return conversation_prompt

async def grid_chat_completion_from_form_data(form_data: dict):
    """Handle Grid model chat completions from form data"""
    try:
        # Extract parameters
        model_name = form_data.get("model", "")
        messages = form_data.get("messages", [])
        max_tokens = form_data.get("max_tokens", 2048)  # Increased default for longer responses
        temperature = form_data.get("temperature", 0.7)
        top_p = form_data.get("top_p", 0.9)
        
        # Convert full conversation history to prompt, filtering out thinking tags
        conversation_prompt = convert_messages_to_prompt(messages)
        
        if not conversation_prompt.strip():
            raise HTTPException(status_code=400, detail="No valid conversation content found")
        
        # Prepare Grid generation request
        grid_request = GridGenerationRequest(
            prompt=conversation_prompt,
            params={
                "max_length": min(max_tokens, 1024),  # Cap at 1024 for optimal performance
                "temperature": temperature,
                "top_p": top_p,
                "top_k": 40,
                "rep_pen": 1.1,
                "rep_pen_range": 256,
                "rep_pen_slope": 0.7,
                "tfs": 1,
                "top_a": 0,
                "typical": 1,
                "singleline": False,
                "frmttriminc": True,
                "frmtrmblln": False,
                "stop_sequence": ["You:", "Human:", "Assistant:", "###"]
            },
            models=[model_name.replace("grid_", "")],  # Remove grid_ prefix
            trusted_workers=False,
            slow_workers=True
        )
        
        # Submit generation request
        t_start = time.time()
        async with aiohttp.ClientSession() as session:
            async with session.post(
                GRID_GENERATE_ENDPOINT, 
                headers=grid_provider.headers, 
                json=grid_request.dict()
            ) as response:
                if response.status != 202:
                    raise HTTPException(status_code=500, detail="Failed to submit Grid generation")
                
                result = await response.json()
                request_id = result.get("id")
                
                if not request_id:
                    raise HTTPException(status_code=500, detail="No request ID received")
                
                # Poll for completion
                max_attempts = 60  # 5 minutes max
                attempts = 0
                
                while attempts < max_attempts:
                    await asyncio.sleep(5)  # Wait 5 seconds
                    attempts += 1
                    
                    status_url = f"{GRID_STATUS_ENDPOINT}/{request_id}"
                    async with session.get(status_url, headers=grid_provider.headers) as status_response:
                        if status_response.status == 200:
                            status_data = await status_response.json()
                            
                            if status_data.get("done"):
                                generations = status_data.get("generations", [])
                                if generations:
                                    generated_text = generations[0].get("text", "")
                                    t_end = time.time()
                                    elapsed_ms = int((t_end - t_start) * 1000)
                                    prompt_tokens = len(conversation_prompt.split())
                                    completion_tokens = len(generated_text.split())
                                    total_tokens = prompt_tokens + completion_tokens
                                    worker_name = _extract_worker_from_status(status_data)
                                    tps = (completion_tokens / max((t_end - t_start), 1e-6)) if completion_tokens > 0 else 0.0
                                    
                                    # Return in OpenAI-compatible format
                                    # This will be processed by the main chat completion pipeline
                                    return {
                                        "id": f"grid_{request_id}",
                                        "object": "chat.completion",
                                        "created": int(time.time()),
                                        "model": model_name,
                                        "choices": [{
                                            "index": 0,
                                            "message": {
                                                "role": "assistant",
                                                "content": generated_text
                                            },
                                            "finish_reason": "stop"
                                        }],
                                        "usage": {
                                            "provider": "grid",
                                            "job_id": request_id,
                                            **({"worker": worker_name} if worker_name else {}),
                                            "elapsed_ms": elapsed_ms,
                                            "tokens_per_sec": tps,
                                            "prompt_tokens": prompt_tokens,
                                            "completion_tokens": completion_tokens,
                                            "total_tokens": total_tokens,
                                        }
                                    }
                                
                                # If no generations, check for errors
                                if status_data.get("faulted"):
                                    raise HTTPException(status_code=500, detail="Grid generation failed")
                                
                        elif status_response.status == 404:
                            raise HTTPException(status_code=500, detail="Grid request not found")
                
                # Timeout
                raise HTTPException(status_code=408, detail="Grid generation timeout")
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in Grid chat completion: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
        
        # Extract parameters
        model_name = body.get("model", "")
        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens", 1024)  # Set to 1024 for optimal performance
        temperature = body.get("temperature", 0.7)
        top_p = body.get("top_p", 0.9)
        
        # Get the last user message
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break
        
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")
        
        # Prepare Grid generation request
        grid_request = GridGenerationRequest(
            prompt=user_message,
            params={
                "max_length": min(max_tokens, 1024),  # Cap at 1024 for optimal performance
                "temperature": temperature,
                "top_p": top_p,
                "top_k": 40,
                "rep_pen": 1.1,
                "rep_pen_range": 256,
                "rep_pen_slope": 0.7,
                "tfs": 1,
                "top_a": 0,
                "typical": 1,
                "singleline": False,
                "frmttriminc": True,
                "frmtrmblln": False,
                "stop_sequence": ["You:", "Human:", "Assistant:", "###"]
            },
            models=[model_name.replace("grid_", "")],  # Remove grid_ prefix
            trusted_workers=False,
            slow_workers=True
        )
        
        # Submit generation request
        t_start = time.time()
        async with aiohttp.ClientSession() as session:
            async with session.post(
                GRID_GENERATE_ENDPOINT, 
                headers=grid_provider.headers, 
                json=grid_request.dict()
            ) as response:
                if response.status != 202:
                    raise HTTPException(status_code=500, detail="Failed to submit Grid generation")
                
                result = await response.json()
                request_id = result.get("id")
                
                if not request_id:
                    raise HTTPException(status_code=500, detail="No request ID received")
                
                # Poll for completion
                max_attempts = 60  # 5 minutes max
                attempts = 0
                
                while attempts < max_attempts:
                    await asyncio.sleep(5)  # Wait 5 seconds
                    attempts += 1
                    
                    status_url = f"{GRID_STATUS_ENDPOINT}/{request_id}"
                    async with session.get(status_url, headers=grid_provider.headers) as status_response:
                        if status_response.status == 200:
                            status_data = await status_response.json()
                            
                            if status_data.get("done"):
                                generations = status_data.get("generations", [])
                                if generations:
                                    generated_text = generations[0].get("text", "")
                                    
                                    # Create OpenAI-compatible response format
                                    # This will be processed by Open WebUI's native pipeline
                                    t_end = time.time()
                                    elapsed_ms = int((t_end - t_start) * 1000)
                                    prompt_tokens = len(user_message.split())
                                    completion_tokens = len(generated_text.split())
                                    total_tokens = prompt_tokens + completion_tokens
                                    worker_name = _extract_worker_from_status(status_data)
                                    tps = (completion_tokens / max((t_end - t_start), 1e-6)) if completion_tokens > 0 else 0.0
                                    response_data = {
                                        "id": f"grid_{request_id}",
                                        "object": "chat.completion",
                                        "created": int(time.time()),
                                        "model": model_name,
                                        "choices": [{
                                            "index": 0,
                                            "message": {
                                                "role": "assistant",
                                                "content": generated_text
                                            },
                                            "finish_reason": "stop"
                                        }],
                                        "usage": {
                                            "provider": "grid",
                                            "job_id": request_id,
                                            **({"worker": worker_name} if worker_name else {}),
                                            "elapsed_ms": elapsed_ms,
                                            "tokens_per_sec": tps,
                                            "prompt_tokens": prompt_tokens,
                                            "completion_tokens": completion_tokens,
                                            "total_tokens": total_tokens,
                                        }
                                    }
                                    
                                    # Process through Open WebUI's native pipeline
                                    from open_webui.utils.middleware import process_chat_response
                                    
                                    # Create a mock response object that process_chat_response expects
                                    class MockResponse:
                                        def __init__(self, data):
                                            self.data = data
                                            self.headers = {"Content-Type": "application/json"}
                                        
                                        def json(self):
                                            return self.data
                                    
                                    mock_response = MockResponse(response_data)
                                    
                                    # Process through native pipeline
                                    return await process_chat_response(
                                        request, mock_response, form_data, user, 
                                        form_data.get("metadata", {}), 
                                        {"id": model_name, "owned_by": "grid"}, 
                                        [], None
                                    )
                                break
                            elif status_data.get("faulted"):
                                raise HTTPException(status_code=500, detail=f"Generation failed: {status_data.get('faulted')}")
                
                raise HTTPException(status_code=408, detail="Generation timed out")
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in Grid chat completion: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Hook into Open WebUI's model loading system
async def get_grid_models_for_openwebui():
    """Get Grid models as plain dicts compatible with Open WebUI model list."""
    try:
        models = await grid_provider.get_models()
        openwebui_models = []

        for model in models:
            openwebui_model = {
                "id": f"grid_{model.name}",
                "name": model.name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "grid",
                "permission": [],
                "root": f"grid_{model.name}",
                "parent": None,
                "meta": {
                    "description": f"Grid model: {model.name}",
                    "queue_length": model.queued,
                    "eta": model.eta,
                    "performance": str(model.performance),
                    "worker_count": model.count,
                },
            }
            openwebui_models.append(openwebui_model)

        return openwebui_models

    except Exception as e:
        logger.error(f"Error getting Grid models for Open WebUI: {e}")
        return []
