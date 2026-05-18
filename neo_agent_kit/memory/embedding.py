"""统一嵌入服务

支持三种嵌入方案:
1. TF-IDF (轻量级兜底，始终可用)
2. sentence-transformers 本地模型
3. DashScope (百炼) 云端 API

设计遵循"优雅降级"原则：优先尝试高质量方案，失败时自动回退。
"""
from __future__ import annotations

import os
import hashlib
import math
from typing import List, Optional


class TFIDFEmbedding:
    """TF-IDF 轻量级嵌入（纯 Python，零依赖）

    始终可用，适合离线部署和快速原型。
    """

    def __init__(self, dimension: int = 256):
        self.dimension = dimension
        self.vocabulary: dict = {}
        self.idf: dict = {}

    def _tokenize(self, text: str) -> List[str]:
        """简单分词（支持中英文混合）"""
        import re
        # 提取中文字符、英文单词、数字
        tokens = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]+|\d+', text.lower())
        return tokens

    def _hash_token(self, token: str) -> int:
        """将 token 哈希到固定维度范围"""
        h = hashlib.md5(token.encode('utf-8')).hexdigest()
        return int(h, 16) % self.dimension

    def encode(self, texts: str | List[str]) -> List[List[float]]:
        """编码文本为向量"""
        if isinstance(texts, str):
            texts = [texts]

        vectors = []
        for text in texts:
            tokens = self._tokenize(text)
            vec = [0.0] * self.dimension

            if not tokens:
                vectors.append(vec)
                continue

            # TF 计算
            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1

            for token, count in tf.items():
                idx = self._hash_token(token)
                tf_val = count / len(tokens)
                vec[idx] += tf_val

            # L2 归一化
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]

            vectors.append(vec)

        return vectors


def _get_sentence_transformers():
    """尝试加载 sentence-transformers"""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except ImportError:
        return None


def create_embedding_model(
    model_type: str = "tfidf",
    model_name: str = "",
    api_key: str = "",
    base_url: str = "",
) -> object:
    """创建嵌入模型（带自动降级）

    Args:
        model_type: 模型类型 (tfidf / local / dashscope)
        model_name: 模型名称
        api_key: API 密钥 (DashScope)
        base_url: API 基础地址

    Returns:
        嵌入模型实例，需实现 encode(texts) -> List[List[float]] 方法
    """

    # 尝试 local (sentence-transformers)
    if model_type == "local":
        ST = _get_sentence_transformers()
        if ST is not None:
            name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
            try:
                print(f"📦 加载本地嵌入模型: {name}")
                model = ST(name)
                return _SentenceTransformerWrapper(model)
            except Exception as e:
                print(f"⚠️ 本地模型加载失败: {e}，降级到 TF-IDF")

    # 尝试 DashScope (百炼)
    if model_type == "dashscope":
        try:
            import dashscope
            dashscope.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
            if dashscope.api_key:
                print(f"☁️ 使用 DashScope 嵌入服务")
                return _DashScopeWrapper(model_name or "text-embedding-v3")
        except ImportError:
            print("⚠️ dashscope 未安装，降级到 TF-IDF")
        except Exception as e:
            print(f"⚠️ DashScope 初始化失败: {e}，降级到 TF-IDF")

    # 默认: TF-IDF (永远可用)
    print("📊 使用 TF-IDF 轻量级嵌入")
    return TFIDFEmbedding()


class _SentenceTransformerWrapper:
    """sentence-transformers 包装器"""

    def __init__(self, model):
        self.model = model

    def encode(self, texts: str | List[str]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()


class _DashScopeWrapper:
    """DashScope 嵌入服务包装器"""

    def __init__(self, model_name: str = "text-embedding-v3"):
        self.model_name = model_name

    def encode(self, texts: str | List[str]) -> List[List[float]]:
        import dashscope
        from dashscope import TextEmbedding

        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []
        for text in texts:
            resp = TextEmbedding.call(
                model=self.model_name,
                input=text,
            )
            if resp.status_code == 200:
                all_embeddings.append(resp.output["embeddings"][0]["embedding"])
            else:
                # 失败时返回零向量
                all_embeddings.append([0.0] * 1024)

        return all_embeddings


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """计算两个向量的余弦相似度"""
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)
