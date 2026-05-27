# Testing and Acceptance

## Automated Checks

Run the test suite from the project root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

The automated suite covers:

- Reddit URL detection for `www.reddit.com`, `reddit.com`, and `old.reddit.com` post links.
- JSON array parsing from direct JSON, fenced markdown blocks, embedded text, and invalid input.
- Review session selection with the 20-card limit, smaller available sets, lower-score priority, and never-reviewed items.
- Score updates for word, phrase, and rule cards.
- The five-user registration limit while allowing existing users to continue.
- Per-user active processing job protection for `queued` and `processing` jobs, plus new job creation after `done`.

## Manual Acceptance Scenario

1. Fill `.env` from `.env.example`.
2. Start the bot:

```powershell
docker compose up --build
```

3. Send `/start` to the bot.
4. Send a Reddit post URL.
5. Check `/status`.
6. Wait for the completion message.
7. Run `/review_words` and press a 1-5 score button.
8. Confirm the next card appears, and the session finishes after the last card.
9. Repeat the same flow for `/review_phrases` and `/review_rules`.

## Acceptance Criteria

The project is ready when:

- `docker compose up --build` starts the bot.
- The bot responds to `/start`.
- No more than five new users are registered.
- Reddit URLs are accepted and create `processing_jobs`.
- The background loop processes queued jobs and stores Reddit text in `raw_text`.
- Each LLM extraction creates an `llm_extraction_jobs` row.
- Words, phrases, and rules are saved to their normalized tables.
- Review commands show cards, score buttons update `current_score`, and sessions complete cleanly.
- One user's review flow does not block other users.
- Heavy Reddit/LLM processing is limited to one active job per user.
- Reddit and LLM errors mark jobs failed without crashing the bot.
