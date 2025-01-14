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


struct TaskManager:
    var tasks: List[Task]

    fn __init__(out self):
        self.tasks = List[Task]()

    fn add_task(mut self, task: Task):
        self.tasks.append(task)

    fn get_task(ref self, index: Int) -> ref [self.tasks] Task:
        # Return a reference to the specific Task origin
        return self.tasks[index]

    fn show_tasks(read self):
        for t in self.tasks:
            print("- ", t[].description)

def main():
    manager = TaskManager()

    manager.add_task(Task("Walk the dog"))
    manager.add_task(Task("Write Mojo 24.6 blog post"))

    # We can fetch a ref to a Task and mutate it in place if needed
    first_task = manager.get_task(0)
    first_task.description = "Walk the dog ASAP and then write the blog post!"

    manager.show_tasks()
    # Output:
    # Walk the dog ASAP and then write the blog post!
    # Write Mojo 24.6 blog post
