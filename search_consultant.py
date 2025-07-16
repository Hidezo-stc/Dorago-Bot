# search_and_rag.py

import pickle
import numpy as np
import faiss
from openai import OpenAI           # ← v1 client
import os
from dotenv import load_dotenv

load_dotenv()

# 1) initialize the new v1 client
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# 2) your existing embedding helper (unchanged)
def get_embedding(text: str, model: str = "text-embedding-3-small") -> list:
    resp = client.embeddings.create(input=[text], model=model)
    return resp.data[0].embedding

# 3) your existing search_consultants (unchanged)…
def search_consultants(query: str,
                       index_path: str = "consultants.index",
                       metadata_path: str = "consultants_meta.pkl",
                       top_k: int = 5):
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    q_vec = np.array(get_embedding(query)).astype("float32").reshape(1, -1)
    _, idxs = index.search(q_vec, top_k)
    return [metadata[i] for i in idxs[0]]

# 4) new rag_answer using v1 Chat Completions
# ─── 要約関数 ───────────────────────────────
def summarize_text(text: str, max_chars: int = 100) -> str:
    """
    OpenAI Chat API で text を約 max_chars 文字に要約して返す。
    """
    prompt = (
        f"以下の文章を日本語でわかりやすく、かつ約{max_chars}文字程度に要約してください。\n\n"
        f"{text}"
    )
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "あなたは優れた要約ライターです。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    return resp.choices[0].message.content.strip()

# ─── RAG＋要約 組み込み例 ──────────────────────
def rag_with_short_summaries(query: str,
                             top_k: int = 3,
                             index_path: str = "consultants.index",
                             meta_path: str = "consultants_meta.pkl") -> str:
    # 1) ベクトル検索
    hits = search_consultants(query, index_path=index_path, metadata_path=meta_path, top_k=top_k)

    # 2) 各ヒットを100字要約
    short_summaries = []
    for h in hits:
        short = summarize_text(h["summary"], max_chars=100)
        short_summaries.append(f"【{h['name']}】 {short}")

    # 3) 回答生成プロンプト
    context = "\n".join(short_summaries)
    messages = [
        {"role": "system", "content": "あなたは社内人材に詳しいアシスタントです。"},
        {"role": "user", "content":
            f"以下の候補について、それぞれ簡潔に紹介してください。\n\n{context}\n\n"
            f"質問: {query}\n\n回答:"
        }
    ]

    resp = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.2
    )
    return resp.choices[0].message.content.strip()

# ─── 使い方 ─────────────────────────────────
if __name__ == "__main__":
    q = "通信業界で新規事業立案経験のある人は？"
    # print(rag_with_short_summaries(q))

    print("=== Hits ===")
    for person in search_consultants(q):
        # print full summary, not summary[:100]
        summary = summarize_text(person['summary'])
        print(f"{person['name']}:\n{summary}\n")
    print("=== RAG Answer ===")
    # print(rag_answer(q))

"""
  git config --global user.email "h-ikari@strategy-tec.com"
  git config --global user.name "Hidezo-stc"
"""