# Ravikumar Buragapu — Engineering Portfolio

Source for [ravikumarburagapu.com](https://ravikumarburagapu.com), a personal portfolio covering AI engineering, platform engineering, reliability, writing, speaking, and open-source research projects.

## Local development

Requires Node.js 22.23.4 or newer.

```sh
npm install
npm run dev
npm run build
npm run preview
```

Astro's file-based routes are in `src/pages/`. The independent research projects are organized in top-level directories; each project has its own README and setup instructions.

## Deployment

The site builds as a static Astro site into `dist/`. `wrangler.jsonc` configures the Cloudflare Workers static-assets deployment and custom-domain routes. Deploy only from an account authorized to manage the configured domain.

## License

Unless a project directory provides its own license, website content and source are published for portfolio use. Individual research projects may have separate licenses; see their `LICENSE` files.
