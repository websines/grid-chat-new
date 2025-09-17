<script>
	import { getContext } from 'svelte';
	const i18n = getContext('i18n');

	import StatusItem from './StatusHistory/StatusItem.svelte';
	export let statusHistory = [];
	export let expand = false;

	let showHistory = true;

	$: if (expand) {
		showHistory = true;
	} else {
		showHistory = false;
	}

	let history = [];
	let displayHistory = [];
	let status = null;

	const hasWaitTime = (entry) => entry?.wait_time !== undefined && entry?.wait_time !== null;

	$: if (history && history.length > 0) {
		status = history.at(-1);
	}

	$: if (JSON.stringify(statusHistory) !== JSON.stringify(history)) {
		history = statusHistory;
	}

	$: displayHistory = (() => {
		if (!history || history.length === 0) {
			return [];
		}
		let lastWaitEntry = null;
		const entries = [];
		for (const entry of history) {
			if (hasWaitTime(entry)) {
				lastWaitEntry = entry;
				continue;
			}
			entries.push(entry);
		}
		if (lastWaitEntry) {
			entries.push(lastWaitEntry);
		}
		return entries;
	})();
</script>

{#if history && history.length > 0}
	{#if status?.hidden !== true}
		<div class="text-sm flex flex-col w-full">
			{#if showHistory}
				<div class="flex flex-row">
					{#if displayHistory.length > 1}
						<div class="w-1 border-r border-gray-50 dark:border-gray-800 mt-3 -mb-2.5" />

						<div class="w-full -translate-x-[7.5px]">
							{#each displayHistory as status, idx}
								{#if idx !== displayHistory.length - 1}
									<div class="flex items-start gap-2 mb-1">
										<div class="pt-3 px-1">
											<span class="relative flex size-2">
												<span
													class="relative inline-flex size-1.5 rounded-full bg-gray-200 dark:bg-gray-700"
												></span>
											</span>
										</div>
										<StatusItem {status} done={true} />
									</div>
								{/if}
							{/each}
						</div>
					{/if}
				</div>
			{/if}

			<button
				class="w-full -translate-x-[3.5px]"
				on:click={() => {
					showHistory = !showHistory;
				}}
			>
				<div class="flex items-start gap-2">
					<div class="pt-3 px-1">
						<span class="relative flex size-2">
							{#if status?.done === false}
								<span
									class="absolute inline-flex h-full w-full animate-ping rounded-full bg-gray-400 dark:bg-gray-700 opacity-75"
								></span>
							{/if}
							<span class="relative inline-flex size-1.5 rounded-full bg-gray-200 dark:bg-gray-700"
							></span>
						</span>
					</div>
					<StatusItem {status} />
				</div>
			</button>
		</div>
	{/if}
{/if}
