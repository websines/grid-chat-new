let accessToken: string | null = null;

export const setAccessToken = (token: string | null | undefined) => {
	accessToken = token ?? null;
};

export const getAccessToken = () => accessToken;
