# Data Specification

## Source
CourtListener REST API — https://www.courtlistener.com/api/rest/v4/

## What to pull
- Endpoint: /opinions/
- Filter: published opinions only (type=010)
- Date range: 2010–2020
- All 13 federal circuits
- Target: ~1,000 opinions total (80–90 per year)

## Fields needed per opinion
- id
- date_filed
- court
- type (majority / dissent / concurrence)
- plain_text
- opinions_cited

## Also pull separately
- Endpoint: /clusters/
- Same date range and courts
- This gives us majority/dissent groupings per case

## Format
Save each year as a separate JSON file:
opinions_2010.json, opinions_2011.json, ... opinions_2020.json

## Where to put it
Push everything to /data/raw/ in this repo

## Priority
Also search for these specific cases by name (circuit split validation set):
[YOU WILL PASTE YOUR CIRCUIT SPLIT LIST HERE]
