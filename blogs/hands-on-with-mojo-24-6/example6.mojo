from collections import Deque

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
    var tasks: Deque[Task]  # Using Deque instead of List

    fn __init__(out self):
        self.tasks = Deque[Task]()

    fn add_task(mut self, task: Task):
        # We can append to the back
        self.tasks.append(task)

    fn add_task_front(mut self, task: Task):
        # or to the front
        self.tasks.appendleft(task)

    fn show_tasks(read self):
        for t in self.tasks:
            print("- ", t[].description)

    @staticmethod
    fn bootstrap_example(out manager: TaskManager):
        manager = TaskManager()
        manager.add_task("Deque-based Task #1")
        manager.add_task_front("Deque-based Task #0")
        return

def main():
    mgr = TaskManager.bootstrap_example()
    mgr.show_tasks()
    # Output:
    # Deque-based Task #0
    # Deque-based Task #1
