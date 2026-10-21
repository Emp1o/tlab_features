# Заморозка теста: контрольные суммы входных файлов

`scripts/run_test.py` и `scripts/run_test_posthoc.py` читают этот файл и сверяют sha256 входов перед запуском. Коммиты предрегистрации, формулировки и результата: `bbf4b42`, `d31cb8a`, `98bbc9a`.

sha256 входных файлов:

| файл | sha256 |
|---|---|
| `data/events.csv` | `3745aa83238b1e6d44d8fda209807899f420084398f94ddf745f3cbcfecbf9e7` |
| `data/category_tree.csv` | `94e865eb0a3d48cbbfe3b79079018dd92509315c88f5fd8d00d0b4b5af434f5b` |
| `data/item_properties_part1.csv` | `30aad5aeca58b2dc27dcc73e1708565f5818e45adb3eb57401f91e87355b0b81` |
| `data/item_properties_part2.csv` | `d5e7d1a91dc40f522aeb596b267e6c87d8aed689a7192d12369cfb165eb987e5` |
| `data/examples.parquet` | `df688362ce67781bdfbfb7caae93e46e846b935966ad8d44f5523faeb3fd14c4` |
| `data/events_clean.parquet` | `9c7681f46a7041653f6602dbe226d15327824c9c0d1a764d498c71bd91f01ad3` |
| `data/feature_candidates.parquet` | `9b824f00e340644d51f82db5b059d135547121f734245cca73e1da41f141ad50` |
| `data/features_manual.parquet` | `8d9f63080f3bf8d58f48f7600f1f26b24b6807b4cf56dde5662af63d23bea21b` |
| `data/item_category.parquet` | `1adc9ff58c05306fe5ac48787a52543967cce3024ba2b2565f767f0dda485c0a` |

## Контрольные суммы автопоиска

sha256 файлов автопоиска:

| файл | sha256 |
|---|---|
| `data/feature_candidates_wsearch.parquet` | `da99b223cbe819f65b2a7b1f49ada41b609b808ef9b7f396a500c48ca8358fe4` |
| `report/window_search.csv` | `33f461a0ce924c5124536fca77000eaa0ecd208ac1eaf1214ff0a9688d0ca466` |
| `report/window_selection_train.csv` | `8a302f39777284559eb2fa6c4d7ecb6325b08e88de9aafb4df85f85b5dfb7159` |
| `report/window_selection_train2.csv` | `df6b0c497dd2789a7b303e67ff9c8ee6dce01273f184c40ad1be9bfbd4d3a7a6` |

