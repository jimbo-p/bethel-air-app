/// <reference path="../.astro/types.d.ts" />

interface ImportMetaEnv {
	readonly PUBLIC_BLOG_MANIFEST_URL?: string;
	readonly PUBLIC_BLOG_BASE_URL?: string;
}

interface ImportMeta {
	readonly env: ImportMetaEnv;
}