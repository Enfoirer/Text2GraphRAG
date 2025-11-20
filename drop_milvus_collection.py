from pymilvus import MilvusClient, MilvusException

client = MilvusClient(uri="http://127.0.0.1:19530")
collection = "cooking_knowledge"

try:
    if client.has_collection(collection):
        client.drop_collection(collection)
        print(f"✅ dropped collection: {collection}")
    else:
        print(f"⚠️ collection {collection} 不存在，无需删除")
except MilvusException as e:
    print(f"删除失败: {e}")
