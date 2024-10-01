"""
@Description  :   TSP File Loader
@Author1      :   Yang Xu, 300342009
@Author2      :   XXX, XXX
@Comment      :   Dev with Python 3.10.0
"""

from tkinter import filedialog


class TSPFile:
    def __init__(self, filepath):
        """
        Initialize the TSP file object with the given file path.

        Parameters:
        - filepath (str): The path to the TSP file.

        Returns:
        - valid (bool): True if the file was successfully loaded and validated, False otherwise.
        """
        self.filepath = filepath
        self.name = None
        self.type = None
        self.comment = None
        self.dimension = None
        self.edge_weight_type = None
        self.node_coords = []
        self.solution = None
        self.total_cost = None

        self.__load()

    def __load(self):
        """
        Load the TSP file and extract the relevant information from the header and node coordinates.
        """
        with open(self.filepath, 'r') as file:
            for line in file:
                # Extract information from the header
                if   line.startswith('NAME'): self.name = line.split(':')[1].strip()
                elif line.startswith('TYPE'): self.type = line.split(':')[1].strip()
                elif line.startswith('COMMENT'):  self.comment = line.split(':')[1].strip()
                elif line.startswith('DIMENSION'): self.dimension = int(line.split(':')[1].strip())
                elif line.startswith('EDGE_WEIGHT_TYPE'): self.edge_weight_type = line.split(':')[1].strip()
                elif line.startswith('NODE_COORD_SECTION'): break
            
            # Read the node coordinates
            for line in file:
                if line.strip() == 'EOF': break

                parts = line.split()
                if len(parts) >= 3:
                    node_id = int(parts[0])
                    x_coord = float(parts[1])
                    y_coord = float(parts[2])
                    self.node_coords.append((node_id, x_coord, y_coord))

    def validate(self):
        """
        Validate the instance variables to ensure that the file is a valid TSP problem and meet the requirements for this assignment.

        Returns:
        - valid (bool): True if the file is a valid TSP problem with EUC_2D edge weights, False otherwise.
        """
        if self.name and self.type and self.dimension and self.edge_weight_type and self.node_coords:
            if self.type == 'TSP' and self.edge_weight_type == 'EUC_2D':    # Only fpcus on TSP and EUC_2D for this assignment
                return True
        return False


class TSPLoader:
    def __init__(self):
        self.tsp_files = []

    def choose_files(self):
        """
        Prompt the user to choose TSP files using a file dialog.
        """
        filepaths = filedialog.askopenfilenames(
            title='Select TSP Files',
            filetypes=[('TSP files', '*.tsp')]
        )
        for filepath in filepaths:
            tsp_file = TSPFile(filepath)
            if tsp_file.validate(): self.tsp_files.append(tsp_file)

    def load_files_from_args(self, filepaths):
        """
        Load TSP files from the command line arguments.
        """
        for filepath in filepaths:
            tsp_file = TSPFile(filepath)
            if tsp_file.validate(): self.tsp_files.append(tsp_file)
