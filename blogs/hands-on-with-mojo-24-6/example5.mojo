@value
struct Task:
    var description: String

    @implicit
    fn __init__(out self, desc: String):
        self.description = desc

    @implicit
    fn __init__(out self, desc: StringLiteral):
        self.description = desc

struct TaskManager:
    var tasks: List[Task]

    fn __init__(out self):
        self.tasks = List[Task]()

    fn add_task(mut self, task: Task):
        self.tasks.append(task)

    fn show_tasks(self):
        for t in self.tasks:
            print("- ", t[].description)

    # Named result: "out manager" is returned directly
    @staticmethod
    fn bootstrap_example(out manager: TaskManager):
        manager = TaskManager()
        manager.add_task("Default Task #1")
        manager.add_task("Default Task #2")
        return  # 'manager' is implied

def main():
    # We can create a TaskManager with default tasks:
    mgr = TaskManager.bootstrap_example()
    mgr.show_tasks()
    # Output:
    # Default Task #1
    # Default Task #2
