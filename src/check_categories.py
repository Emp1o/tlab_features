import duckdb

duckdb.sql("""
    SELECT count(*) AS items,
           count(*) FILTER (WHERE n_cat > 1) AS items_with_changes
    FROM (SELECT item_id, count(DISTINCT category_id) AS n_cat
          FROM 'data/item_category.parquet' GROUP BY item_id)
""").show()