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

    fn show_tasks(read self):
        for t in self.tasks:
            print("- ", t[].description)

def main():
    manager = TaskManager()

    # Because of @implicit, we can now do this:
    manager.add_task(String("Walk the dog"))  # Implicitly converted String -> Task
    manager.add_task("Write Mojo 24.6 blog post")

    manager.show_tasks()
    # Output:
    # Walk the dog
    # Write Mojo 24.6 blog post
