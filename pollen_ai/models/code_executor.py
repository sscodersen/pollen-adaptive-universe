import subprocess

class CodeExecutor:
    def execute_code(self, code):
        try:
            result = subprocess.run(code, shell=True, capture_output=True, text=True)
            return result.stdout, result.stderr
        except Exception as e:
            return None, str(e)