# Project instructions

Engineering context for this repository (structure, runtime environment, testing, Mojo port status, upgrade policy) is in [AGENTS.md](AGENTS.md). Read it before changing code. The rules below are about how to write prose in this repo, and they apply to every agent and every session.

## Markdown in this repo

Doug edits markdown here using RStudio's *Visual* mode, which rewrites the whole file to canonical form on every save. Anything written in a form Visual mode would reformat comes back on the next open as a diff nobody authored.

- **Do not hard-wrap.** Write each paragraph as a single line. Never add an `editor_options: markdown: wrap:` block to a file, and do not reinstate one that was removed.
- **No pipe tables.** Use bullet lists instead.
- **Keep inline code spans short.**
- **Escape markdown-significant characters in prose, the way Visual mode will.** Write `-\>` not `->`, and `\~` not `~`. Same for a literal `\<`, `\>` or `\|` in running text.
- **Do not escape inside code.** Backtick spans and fenced blocks are left verbatim, so `` `P = 2 -> 3` `` stays as it is. Genuine markdown syntax is untouched too: `~~strikethrough~~` and a leading `>` blockquote marker stay as they are.
- **Re-read a file immediately before and after editing it**, since it may be open in the RStudio editor. If the result looks corrupted or duplicated, rewrite the whole file with Write rather than patching with Edit.
- **No em-dashes** anywhere, including commit messages. Use a plain double hyphen (`--`).

Wrapping "to N columns" is not a fix. It was tried and measured, and RStudio's canonicalisation cannot be reproduced outside RStudio, so no wrap width converges. Do not propose one.

`MarkdownWrap: None` in `sjSDM.Rproj` is what enforces this on the editor side. Both halves are load-bearing: the `.Rproj` setting stops RStudio reflowing on save, and this file stops an agent hard-wrapping in the first place.

## Tables in R Markdown

The no-pipe-tables rule covers markdown you write by hand. It does not cover tables produced by code. When a table's numbers come from a fit, a benchmark, or any computation, generate it inside a chunk with `knitr::kable()` rather than transcribing the values into markdown. Chunk output is not part of the source Visual mode canonicalises, so those tables are stable, and they cannot drift out of date the way a transcribed one does.

## Numbers in prose

Do not write expected values from memory or estimation. If a document states that a fit recovers a coefficient to some tolerance, or that a benchmark shows some speedup, run it and copy the number. When a claim and its own rendered output disagree, the claim is what is wrong.
