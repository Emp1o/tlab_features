import duckdb

con = duckdb.connect()
con.sql("CREATE VIEW events AS SELECT * FROM read_csv_auto('data/events.csv')")

con.sql("SELECT event, count(*) AS n FROM events GROUP BY event ORDER BY n DESC").show()

con.sql("""
    SELECT to_timestamp(min(timestamp) / 1000) AS first,
           to_timestamp(max(timestamp) / 1000) AS last,
           count(*) AS events,
           count(DISTINCT visitorid) AS users,
           count(DISTINCT itemid) AS items
    FROM events
""").show()

con.sql("""
    SELECT quantile_cont(n, [0.5, 0.9, 0.99, 0.999]) AS quantiles,
           max(n) AS max_n
    FROM (SELECT visitorid, count(*) AS n FROM events GROUP BY visitorid)
""").show()

con.sql("""
    SELECT visitorid, count(*) AS n,
           count(*) FILTER (WHERE event = 'transaction') AS buys
    FROM events GROUP BY visitorid ORDER BY n DESC LIMIT 10
""").show()