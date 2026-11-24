# Ravikumar Buragapu — Engineering Portfolio

Source for [ravikumarburagapu.com](https://ravikumarburagapu.com), a personal portfolio covering AI engineering, platform engineering, reliability, writing, speaking, and open-source research projects.

## Local development

Requires Node.js 22.23.4 or newer.

```sh
npm install
npm run dev -- --background
npx astro dev status
npx astro dev logs
npx astro dev stop
npm run build
npm run preview
```

After restarting VS Code or your system, open the repository in VS Code and check `npx astro dev status` in a terminal. If the server is not running, start it with `npm run dev -- --background`, then open [http://localhost:4321](http://localhost:4321). The background server runs independently of the terminal; use the status, logs, and stop commands above to manage it. If dependencies are not installed, run `npm install` first.

Astro's file-based routes are in `src/pages/`. The independent research projects are organized in top-level directories; each project has its own README and setup instructions.

## Deployment

The site builds as a static Astro site into `dist/`. `wrangler.jsonc` configures the Cloudflare Workers static-assets deployment and custom-domain routes. Deploy only from an account authorized to manage the configured domain.

## License

Unless a project directory provides its own license, website content and source are published for portfolio use. Individual research projects may have separate licenses; see their `LICENSE` files.
