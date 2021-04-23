# Notes

This is Jan Meppe's collection of personal notes on coding, statistics, machine learning, and more. 

Shamelessly copied from [Chris Albon](https://chrisalbon.com/) because I believe that [knowledge work should accrete](https://notes.andymatuschak.org/Knowledge_work_should_accrete).

## Add front-matter manually

Note that you **must add a `yaml` like frontmatter** to your post otherwise it doesn't show up.

For example

```
---
title: "Create simulated data for clustering"
author: "Jan Meppe"
date: 2021-04-09
description: "Create simulated data for clustering"
type: technical_note
draft: false
---
```

## How the website works

* The order of the pages is hardcoded in `themes/berbera/layouts/index.html`

## Overview

The master record of a note is either a Jupyter Notebook or a Markdown file. These files are in the `content` folder. The website HTML is contained in the `docs` folder.

## Full Deploy Procedure

1. Run `python make.py` to convert the Jupyter Notebooks and associated images into Markdown files.
2. Run `hugo` to convert the Markdown files into HTML pages.
3. Run `git add -A` 
4. Run `git commit -m "commit message"`
5. Run `git push`

## Markdown Head Metadata Example

```
---
title: "Give Table An Alias"
author: "Chris Albon"
date: 2019-01-28T00:00:00-07:00
description: "Give a table an alias in Snowflake using SQL."
type: technical_note
draft: false
---
```

## Useful Aliases

To reduce the barriers to publishing a new note as much as possible, here are some useful aliases for your `.bash_profile`:

```
# Notes Project

# Go to Notes folder
alias nn='cd /Users/chrisalbon/dropbox/cra/projects/notes'

# Go to Notes folder and open Jupyter Notebook
alias njn='cd /Users/chrisalbon/dropbox/cra/projects/notes && jupyter notebook'

# For me
alias njn='cd ~/Documents/Projects/notes-1 && jupyter notebook'

# Launch in Hugo server of Notes site
alias nhs='cd /Users/chrisalbon/dropbox/cra/projects/notes && hugo server'

# Publish a new note
alias nnn='cd /Users/chrisalbon/dropbox/cra/projects/notes && git pull && hugo && git add -A && git commit -m "made changes" && gp && git push'
```

Note that when you run `nnn` you might be prompted for an application password. You can get that / generate that from GitHub.com in account settings.

## To Do

- Make `make.py` faster 
- Change links in header/footer etc
- Change google analytics