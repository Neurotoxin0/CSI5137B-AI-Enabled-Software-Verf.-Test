import re, os


Path = (os.path.split(os.path.realpath(__file__))[0] + "/").replace("\\\\", "/").replace("\\", "/")
os.chdir(Path)


with open('main.log', 'r', encoding='utf-8') as file: content = file.read()
pattern = re.compile(r'Solution: \[.*?\] \n Solution Validation: \(.*?\)', re.DOTALL)
modified_content = re.sub(pattern, '', content)
with open('main_without_solution.log', 'w', encoding='utf-8') as file:  file.write(modified_content)
