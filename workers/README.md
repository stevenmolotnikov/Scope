# Workers

### Deploying the workers to Cloudflare

From within the workers dir:

```
pn wrangler login
pn wrangler secret put OPENROUTER_API_KEY --env=""
pn wrangler deploy
```