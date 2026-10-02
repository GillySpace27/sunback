# Rotating the dispatcher's GitHub token

The 20-minute cadence comes from EventBridge rule `sun-reducer-20min` invoking
Lambda `sun-reducer-dispatcher`, which calls GitHub's workflow-dispatch API for
`GitCloudRunHourly.yml` with a token stored in Secrets Manager as
`github-actions-dispatch-token`. When the token expires the dispatch fails and
the Sun updates only from the hourly fallback schedule, which runs when the
newest image is more than 30 minutes old. Gilly does every step below himself.
Never paste the token into a Claude session, a file, an issue or a commit.

## When

- `python infra/pat_expiry.py` exits 1 (14 days or less left, or expired).
- The secret has no `pat-expires` tag (exit 3): the expiry is unknown, so
  rotate soon rather than wait.

## 1. Create the new token on GitHub

GitHub, Settings, Developer settings, Personal access tokens, Fine-grained
tokens, Generate new token:

- Resource owner: GillySpace27.
- Repository access: Only select repositories, `GillySpace27/sunback`.
- Repository permissions: Actions, Read and write (the workflow-dispatch API
  needs it); Metadata, Read (added automatically).
- Expiration: pick a date and note it; the script asks for it.

Copy the token once; GitHub shows it only on this page.

## 2. Store it, tag it, prove it (your own terminal, your AWS profile)

```bash
cd ~/vscode/sunback
python infra/rotate_dispatch_pat.py
```

It asks for the expiry date, then the token at a hidden prompt, prints a plan
and changes nothing until you type `rotate`. Then it stores the token as the new
current version (the old one stays as `AWSPREVIOUS`), tags the secret
`pat-expires=<date>`, invokes the dispatcher once and waits up to 5 minutes for
a new `workflow_dispatch` run. The test dispatch starts one real production
reducer run, the same as any 20-minute dispatch.

Expected last line: `OK: run https://github.com/GillySpace27/sunback/actions/runs/<id> created <time> (after <time>)`.

## 3. Afterwards

- Revoke the old token on GitHub (Settings, Developer settings, Personal access
  tokens) only after step 2 printed `OK`. The old value still sits in the secret
  as `AWSPREVIOUS`; once revoked on GitHub it is useless.
- `python infra/pat_expiry.py` now prints `OK: dispatcher PAT expires in <n> day(s) (<date>)`.
- Re-snapshot and open a PR for the changed tag:
  `python infra/snapshot.py --out infra/live && python infra/diff.py`, then copy
  `infra/live/secret-github-actions-dispatch-token.json` into `infra/declared/`.

## If it fails

The script prints the exact command that moves `AWSCURRENT` back to the previous
version (`aws secretsmanager update-secret-version-stage ...`). Run it only if
the old token is still valid on GitHub. Read the dispatcher's log with
`aws logs tail /aws/lambda/sun-reducer-dispatcher --region us-east-2 --since 15m`.
The hourly fallback keeps the Sun updating meanwhile.
