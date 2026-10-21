from pathlib import Path


class PlannerAgent:

    def create_plan(self, application_folder: Path):

        print("Planner Agent")
        print("----------------------")

        files = list(application_folder.glob("*"))

        print(f"Found {len(files)} documents\n")

        tasks = [
            "Classify Documents",
            "Verify Identity",
            "Verify Employment",
            "Analyze Bank Statements",
            "Calculate Affordability",
            "Generate Underwriting Summary"
        ]

        for index, task in enumerate(tasks, start=1):
            print(f"{index}. {task}")

        return {
            "documents_found": len(files),
            "tasks": tasks
        }