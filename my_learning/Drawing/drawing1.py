import turtle

scr = turtle.Screen()
scr.title("Drawing by apsara")
scr.bgcolor("yellow")
scr.setup(500, 500)

pen = turtle.Turtle()
pen.color("brown")

pen.shape("turtle")
pen.pensize(15)
pen.fillcolor("pink")
pen.begin_fill()
for i in range(4):
    pen.fd(200)
    pen.rt(90)
pen.end_fill()

pen.fillcolor("pink")
pen.begin_fill()
for i in range(4):
    pen.fd(100)
    pen.rt(110)
pen.end_fill()

pen.fillcolor("pink")
pen.begin_fill()
for i in range(4):
    pen.fd(-100)
    pen.rt(90)
pen.end_fill()

pen.fillcolor("black")
pen.begin_fill()
for i in range(4):
    pen.fd(-100)
    pen.rt(90)
pen.end_fill()

pen.fillcolor("red")
pen.begin_fill()
for i in range(4):
    pen.fd(-900)
    pen.rt(40)
pen.end_fill()

