@value
struct Task:
    var description: String

    fn __init__(out self, desc: String):
        # 'out self' indicates we are constructing a fresh Task
        self.description = desc

struct TaskManager:
    var tasks: List[Task]

    fn __init__(out self):
        # Another 'out self' constructor, sets up an empty list of tasks
        self.tasks = List[Task]()

    fn add_task(mut self, task: Task):
        self.tasks.append(task)

    fn show_tasks(read self):
        for t in self.tasks:
            print("- ", t[].description)

def main():
    # Create a new TaskManager
    manager = TaskManager()

    # Add tasks
    manager.add_task(Task("Walk the dog"))
    manager.add_task(Task("Write Mojo 24.6 blog post"))

    manager.show_tasks()
    # Output:
    # Buy groceries
    # Write Mojo 24.6 blog post
