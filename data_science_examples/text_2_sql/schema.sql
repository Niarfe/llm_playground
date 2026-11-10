-- ---------------------------------------------------------------------------
-- A small billing database. Four tables, 16 invoices, hand-written.
--
-- WHY HAND-WRITTEN
--   The obvious shortcut is to ask a model to generate this. Don't. The
--   model under test is the thing generating SQL against this schema, and
--   if it also invented the rows, then any question it answers correctly
--   might only mean it remembered its own output. The eval would be
--   confirming itself.
--
--   So every row below was typed by a person. It is small enough to check
--   by eye, which is the only reason the reference answers in questions.py
--   can be trusted at all.
--
-- WHAT IS DELIBERATELY AWKWARD
--   Real data is not tidy, and three bits of untidiness here are on purpose.
--   Each one breaks a different naive query, and each is called out at the
--   point where it bites.
--
--     1. There is no "overdue" column. Lateness has to be derived by
--        comparing a payment date to a due date.
--     2. Invoice 4475 is still 'outstanding' but has a payment against it.
--        It was paid in part. So status and payments disagree, and "how
--        much have we collected" has a different answer from "how much
--        have we invoiced".
--     3. Invoice 4480 was written off. It is neither paid nor outstanding,
--        so any query that assumes two states gets it wrong.
--
--   Invoices 4471 and 4472 are the same two from examples/05a_rag.py, where
--   semantic search ranked 4472 above 4471 when asked which was paid. SQL
--   does not have that problem. That contrast is the point of putting this
--   example after that one.
--
-- Money is in cents, as integers. Floats do not belong in a ledger.
-- ---------------------------------------------------------------------------

DROP TABLE IF EXISTS line_items;
DROP TABLE IF EXISTS payments;
DROP TABLE IF EXISTS invoices;
DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    id          INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL,
    city        TEXT    NOT NULL,
    signed_up   TEXT    NOT NULL   -- ISO date, YYYY-MM-DD
);

CREATE TABLE invoices (
    id            INTEGER PRIMARY KEY,
    customer_id   INTEGER NOT NULL REFERENCES customers(id),
    issued        TEXT    NOT NULL,  -- ISO date
    due           TEXT    NOT NULL,  -- ISO date
    amount_cents  INTEGER NOT NULL,
    status        TEXT    NOT NULL   -- 'paid' | 'outstanding' | 'written_off'
);

CREATE TABLE payments (
    id            INTEGER PRIMARY KEY,
    invoice_id    INTEGER NOT NULL REFERENCES invoices(id),
    paid_on       TEXT    NOT NULL,  -- ISO date
    amount_cents  INTEGER NOT NULL,
    method        TEXT    NOT NULL   -- 'ach' | 'wire' | 'check' | 'card'
);

CREATE TABLE line_items (
    id                INTEGER PRIMARY KEY,
    invoice_id        INTEGER NOT NULL REFERENCES invoices(id),
    description       TEXT    NOT NULL,
    qty               INTEGER NOT NULL,
    unit_price_cents  INTEGER NOT NULL
);

-- --------------------------------------------------------------------------
-- customers
-- --------------------------------------------------------------------------
INSERT INTO customers (id, name, city, signed_up) VALUES
    (1, 'Alder Freight',     'Richmond',        '2024-03-11'),
    (2, 'Bellweather Press', 'Charlottesville', '2024-05-02'),
    (3, 'Carrow Analytics',  'Richmond',        '2025-01-20'),
    (4, 'Dunmore Hardware',  'Roanoke',         '2023-11-04'),
    (5, 'Eastlight Media',   'Charlottesville', '2025-06-15'),
    (6, 'Fenwick Labs',      'Blacksburg',      '2025-09-01');

-- --------------------------------------------------------------------------
-- invoices
-- --------------------------------------------------------------------------
INSERT INTO invoices (id, customer_id, issued, due, amount_cents, status) VALUES
    (4465, 1, '2026-07-02', '2026-08-01', 125000, 'paid'),
    (4466, 2, '2026-07-09', '2026-08-08',  48000, 'paid'),
    (4467, 1, '2026-07-15', '2026-08-14',  32500, 'paid'),
    (4468, 3, '2026-07-21', '2026-08-20', 275000, 'paid'),
    (4469, 4, '2026-08-03', '2026-09-02',  61000, 'paid'),
    (4470, 2, '2026-08-11', '2026-09-10',  48000, 'paid'),
    (4471, 5, '2026-09-14', '2026-10-14',  94500, 'paid'),
    (4472, 5, '2026-10-01', '2026-10-31', 112000, 'outstanding'),
    (4473, 3, '2026-09-22', '2026-10-22', 275000, 'paid'),
    (4474, 6, '2026-09-28', '2026-10-28',  18000, 'paid'),
    (4475, 1, '2026-10-05', '2026-11-04', 143000, 'outstanding'),
    (4476, 4, '2026-10-12', '2026-11-11',   7250, 'outstanding'),
    (4477, 2, '2026-10-19', '2026-11-18',  48000, 'outstanding'),
    (4478, 6, '2026-10-26', '2026-11-25',  36000, 'outstanding'),
    (4479, 3, '2026-08-30', '2026-09-29', 275000, 'paid'),
    (4480, 4, '2026-06-10', '2026-07-10',  52000, 'written_off');

-- --------------------------------------------------------------------------
-- payments
--
-- 4466, 4469 and 4473 were paid after their due date. Nothing in the schema
-- says so. Only the dates do.
--
-- Payment 11 is the partial one: 500.00 against invoice 4475, which is why
-- that invoice is still marked outstanding.
-- --------------------------------------------------------------------------
INSERT INTO payments (id, invoice_id, paid_on, amount_cents, method) VALUES
    ( 1, 4465, '2026-07-28', 125000, 'ach'),
    ( 2, 4466, '2026-08-20',  48000, 'check'),
    ( 3, 4467, '2026-08-10',  32500, 'ach'),
    ( 4, 4468, '2026-08-19', 275000, 'wire'),
    ( 5, 4469, '2026-09-15',  61000, 'check'),
    ( 6, 4470, '2026-09-04',  48000, 'check'),
    ( 7, 4471, '2026-10-02',  94500, 'ach'),
    ( 8, 4473, '2026-11-03', 275000, 'wire'),
    ( 9, 4474, '2026-10-20',  18000, 'card'),
    (10, 4479, '2026-09-25', 275000, 'wire'),
    (11, 4475, '2026-11-06',  50000, 'ach');

-- --------------------------------------------------------------------------
-- line_items
--
-- Every invoice has items that sum exactly to its amount_cents. That is a
-- property worth knowing, because it means a question about revenue can be
-- answered from either table and should agree. When it stops agreeing in a
-- real database, something is wrong upstream.
-- --------------------------------------------------------------------------
INSERT INTO line_items (id, invoice_id, description, qty, unit_price_cents) VALUES
    ( 1, 4465, 'Freight audit, Q2',          1, 125000),
    ( 2, 4466, 'Monthly retainer',           1,  48000),
    ( 3, 4467, 'Route optimization report',  1,  32500),
    ( 4, 4468, 'Data warehouse migration',   1, 275000),
    ( 5, 4469, 'Inventory sync, phase 1',    1,  61000),
    ( 6, 4470, 'Monthly retainer',           1,  48000),
    ( 7, 4471, 'Campaign dashboard build',   1,  85000),
    ( 8, 4471, 'Rush fee',                   1,   9500),
    ( 9, 4472, 'Attribution model, phase 1', 1, 112000),
    (10, 4473, 'Data warehouse migration',   1, 275000),
    (11, 4474, 'Lab notebook template',      2,   9000),
    (12, 4475, 'Freight audit, Q3',          1, 118000),
    (13, 4475, 'Expedited delivery',         1,  25000),
    (14, 4476, 'Replacement sensor',         5,   1450),
    (15, 4477, 'Monthly retainer',           1,  48000),
    (16, 4478, 'Assay data pipeline',        1,  36000),
    (17, 4479, 'Data warehouse migration',   1, 275000),
    (18, 4480, 'Point of sale integration',  1,  52000);
