# Test LoCoMo on Graphiti

### 一、Ingestion

仿照 Zep 的数据插入格式：

```python
await zep.graph.add(
	data=msg.get('speaker') +': ' + msg.get('text') + img_description,
	type='message',
	created_at=iso_date,
	group_id=group_id,
)
```

其中

```python
session_date = conversation.get(f'session_{session_idx}_date_time') + ' UTC'
date_format = '%I:%M %p on %d %B, %Y UTC'
date_string = datetime.strptime(session_date, date_format).replace(tzinfo=timezone.utc)
iso_date = date_string.isoformat()

blip_caption = msg.get('blip_captions')
img_description = f'(description of attached image: {blip_caption})' if blip_caption is not None else ''
```

因此 **Graphiti 数据按如下方式插入**：

```python
await graphiti.add_episode(
    name=f'conv_{owner_id}_turn_{dia_id}',
    episode_body=episode_body,
    source=EpisodeType.text,
    source_description='locomo10_dialogue_turn',
    reference_time=reference_time,
    group_id=owner_id,
)
```

其中

```python
blip_caption = turn.get('blip_captions')
img_description = f" (description of attached image: {blip_caption})" if blip_caption else ''
episode_body = f"{turn.get('speaker')}: {turn.get('text')}{img_description}"

datetime_key = f"{session_key}_date_time"
datetime_str = conversation_content.get(datetime_key)
session_date_with_utc = datetime_str + ' UTC'
date_format = '%I:%M %p on %d %B, %Y UTC'
reference_time = datetime.strptime(session_date_with_utc, date_format).replace(tzinfo=timezone.utc)
# 时间表示与 Zep 不太一样， Graphiti 就要 datetime 类型的，少一步转化
```

### 二、Search

采取与 Zep 同样的 search 方法（RRF节点重排＋交叉编码边重排）：

```python
search_results = await asyncio.gather(
    zep.graph.search(query=query, group_id=group_id, scope='nodes', reranker='rrf', limit=20),
    zep.graph.search(query=query, group_id=group_id, scope='edges', reranker='cross_encoder', limit=20))

nodes = search_results[0].nodes
edges = search_results[1].edges

context = compose_search_context(edges, nodes)
duration_ms = (time() - start) * 1000

zep_search_results[group_id].append({'context': context, 'duration_ms': duration_ms})
```

因此 **Graphiti 按如下方法查找**：

```python
# --- 核心修改：使用 Graphiti 的方法进行并发搜索 ---
# 1. 准备搜索配置
edge_search_config = EDGE_HYBRID_SEARCH_CROSS_ENCODER.model_copy(deep=True)
edge_search_config.limit = 20
node_search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
node_search_config.limit = 20

# 2. 使用 asyncio.gather 并发执行边搜索和节点搜索
search_results = await asyncio.gather(
    # 边 (Facts) 搜索
    graphiti.search(
        query=query, 
        config=edge_search_config,
        group_ids=[owner_id], # 注意：这里是 group_ids 列表
    ),
    # 节点 (Entities) 搜索
    graphiti._search(
        query=query,
        config=node_search_config,
        group_ids=[owner_id] # 同样传入 group_ids
    )
)

# 3. 解析并发搜索的结果
edges = search_results[0]  # graphiti.search 直接返回 list[EntityEdge]
nodes = search_results[1].nodes # graphiti._search 返回一个带 .nodes 属性的对象
# ----------------------------------------------------

context = compose_search_context(edges, nodes)
duration_ms = (time() - start) * 1000

graphiti_search_results[owner_id].append({'context': context, 'duration_ms': duration_ms, 'question': query})
```

### 三、Responses and Evaluations

与 Zep 相同，不再赘述
