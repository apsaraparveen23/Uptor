import sqlite3 as sql


connection=sql.connect('uptor_203.db')
create_table = """create table course('course_id',int,'course_name'text)"""

# create_table = """CREATE TABLE course (
#     course_id INTEGER,
#     course_name TEXT
# )"""


query_execution=connection.execute(create_table)
connection.commit()
connection.close()