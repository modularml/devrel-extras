from utils.variant import Variant

@value
struct Quit(CollectionElement):
	fn display(self) -> String:
		return "Quit"

@value
struct Move(CollectionElement):
	var x: Int
	var y: Int

	fn display(self) -> String:
		return "Move" + "(" + str(self.x) + ", " + str(self.y) + ")"

alias Message = Variant[Quit, Move]

fn visitor(msg: Message) raises -> String:
	if msg.isa[Quit]():
		return msg.get[Quit]().display()
	elif msg.isa[Move]():
		return msg.get[Move]().display()
	else:
		return Error("Unknown variant")

fn main() raises:
	var m1 = Message(Quit())
	var m2 = Message(Move(1, 2))
	print(visitor(m1))
	print(visitor(m2))
