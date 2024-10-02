"""
@Description  :   TSP File Loader
@Author1      :   Yang Xu, 300342009
@Author2      :   Peizhou Zhang, 300400642
@Comment      :   Dev with Python 3.10.0
"""

from tkinter import filedialog


class TSPFile:
    """
    A class to represent a TSP file and extract the relevant information from the header and node coordinates.

    Attributes:
    - filepath (str): The path to the TSP file.
    - name (str): The name of the TSP problem.
    - type (str): The type of the TSP problem.
    - comment (str): A comment about the TSP problem.
    - dimension (int): The number of nodes in the TSP problem.
    - edge_weight_type (str): The type of edge weights in the TSP problem.
    - node_coords (list): A list of tuples containing the node ID and coordinates.
    - solution (list): A list of node IDs representing the solution path.
    - total_cost (float): The total cost of the solution path.

    Methods:
    - validate(): Validate the instance variables to ensure that the file is a valid TSP problem and meet the requirements for this assignment.
    """
    
    def __init__(self, filepath) -> None:
        """
        Initialize the TSP file object with the given file path.

        Parameters:
        - filepath (str): The path to the TSP file.

        Returns:
        - None
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

    
    def __load(self) -> None:
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

    
    def validate(self) -> bool:
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
    """
    A class to load TSP files from the command line arguments or prompt the user to choose files.

    Attributes:
    - tsp_files (list): A list of TSPFile objects representing the loaded TSP files.

    Methods:
    - choose_files(): Prompt the user to choose TSP files using a file dialog.
    - load_files_from_args(filepaths): Load TSP files from the command line arguments.
    """
    
    def __init__(self) -> None:
        self.tsp_files = []


    def choose_files(self) -> None:
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


    def load_files_from_args(self, filepaths) -> None:
        """
        Load TSP files from the command line arguments.
        """
        for filepath in filepaths:
            tsp_file = TSPFile(filepath)
            if tsp_file.validate(): self.tsp_files.append(tsp_file)
