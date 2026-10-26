# Judge video — ≤5 min shot list (team must appear)

0:00-0:20 team + problem (30s of missed risky lifts = injuries/cost)
0:20-1:00 architecture diagram (passive OpenCV → trigger → Strands/Bedrock → S3/review/discard)
1:00-2:30 LIVE demo: `./watch` on SAFELIFT clip; overlay shows trunk°+score; trigger fires; agent calls VLM; STORE lands in S3; show `/ui` trace
2:30-3:30 the TRACE: scroll `trace.jsonl`, narrate one row chain (OpenCV 0.81 → VLM 0.91 → STORE vs would_ignore)
3:30-4:15 eval numbers (P/R vs LI bands, latency, cost/event) + failure case clip (occlusion → human review)
4:15-5:00 limits, responsible use, repo+endpoint links. End card.
