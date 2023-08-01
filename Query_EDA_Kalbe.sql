-- Query 1
-- Berapa rata-rata umur customer jika dilihat dari marital statusnya?
SELECT 
	marital_status AS "Marital Status",
	AVG(age) AS "Age Average"
FROM customer
GROUP BY 1;

-- Query 2
-- Berapa rata-rata umur customer jika dilihat dari gender-nya?
SELECT
	gender AS "Gender",	
	AVG(age) AS "Age Average"
FROM customer
GROUP BY 1;

-- Query 3
-- Tentukan nama store dengan total quantity terbanyak!
SELECT
	s.store_name AS "Store Name",
	SUM(t.qty) AS "Total Quantity"
FROM transaction AS t
LEFT JOIN store AS s
ON t.store_id = s.store_id
GROUP BY 1
LIMIT 1;

-- Query 4
-- Tentukan nama produk terlaris dengan total amount terbanyak!
SELECT
	p.product_name AS "Product Name",
	SUM(t.total_amount) AS "Total Amount"
FROM transaction AS t
LEFT JOIN product AS p
ON t.product_id = p.product_id
GROUP BY 1
LIMIT 1;