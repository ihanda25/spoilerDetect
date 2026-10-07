# Locked Qwen-alone versus hybrid RAG

100 unchanged inputs, frozen prompt/model/settings and locked AI labels. Reuse alone outputs; run only RAG. Local persisted all-MiniLM-L6-v2 vector index plus BM25, title filter, RRF k60, top3 plot chunks. No fine-tuning or purchases. Groq Free-plan requests paced at >=8s with measured token headroom and bounded retries. Every answer saved; rerun --run to resume. Binary metrics exclude 17 unresolved labels; report evidence validation separately from raw labels. Source corpus is incomplete, notably Firefly comic revelations.
