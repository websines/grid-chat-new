<script>
	import { getAccessToken } from '$lib/utils/tokenStore';
	import { onMount } from 'svelte';
	import { config, models, settings } from '$lib/stores';
	import { getModels } from '$lib/apis';
	import Models from '$lib/components/workspace/Models.svelte';

	onMount(async () => {
		await Promise.all([
			(async () => {
				models.set(
					await getModels(
						getAccessToken(),
						$config?.features?.enable_direct_connections && ($settings?.directConnections ?? null)
					)
				);
			})()
		]);
	});
</script>

{#if $models !== null}
	<Models />
{/if}
