<script lang="ts">
  export let size = 14; // px
  export let className = '';
  // Static path served by SvelteKit: place aipg-logo.png under grid-chat-new/static/branding/
  const src = '/branding/aipg-logo.png';
  let errored = false;
</script>

{#if !errored}
  <img
    src={src}
    alt="AIPG"
    width={size}
    height={size}
    class={`inline-block align-middle opacity-80 ${className}`}
    on:error={() => (errored = true)}
    loading="lazy"
    decoding="async"
  />
{:else}
  <span class={`inline-block align-middle text-[10px] font-medium ${className}`}>AIPG</span>
{/if}

