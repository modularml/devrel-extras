@value
struct Task:
    var description: String

    @implicit
    fn __init__(out self, desc: String):
        # Opt-in to implicit conversion from String → Task
        self.description = desc

    @implicit
    fn __init__(out self, desc: StringLiteral):
        # Opt-in to implicit conversion from StringLiteral → Task
        self.description = desc


fn pick_longer(ref t1: Task, ref t2: Task) -> ref [t1, t2] Task:
    return t1 if len(t1.description) >= len(t2.description) else t2


struct TaskManager:
    var tasks: List[Task]

    fn __init__(out self):
        self.tasks = List[Task]()

    fn add_task(mut self, task: Task):
        self.tasks.append(task)

    fn get_task(ref self, index: Int) -> ref [self.tasks] Task:
        # Return a reference to the specific Task origin
        return self.tasks[index]

    fn show_tasks(self):
        for t in self.tasks:
            print("- ", t[].description)

def main():
    manager = TaskManager()

    manager.add_task("Walk the dog")
    manager.add_task("Write Mojo 24.6 blog post")
    # pick_longer(manager.get_task(0), manager.get_task(1))
    first_task = manager.get_task(0)
    second_task = manager.get_task(1)
    _ = pick_longer(first_task, second_task)

    # We can fetch a ref to a Task and mutate it in place if needed
    #first_task = manager.get_task(0)
    manager.get_task(0).description = "Walk the dog ASAP and then write the blog post!"

    manager.show_tasks()

    # longer = pick_longer(first_task, manager.get_task(1))
    # print("Longer task: ", longer.description)

    # Output:
    # Longer task: Walk the dog ASAP and then write the blog post!

    # Note that the following is not allowed anymore and raises an error in 24.6
    # with `Argument exclusivity` check that enforces strict aliasing rules:
    # pick_longer(manager.get_task(0), manager.get_task(1))
    # error: argument of 'pick_longer' call allows writing a memory location previously writable through another aliased argument
