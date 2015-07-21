package de.tu_berlin.dima.impro3;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by felix on 30.06.15.
 */
public class SPTree {

	// Constructs cell
	public class Cell{
		private int dimension;
		private double [] corner;
		private double [] width;
		
		
		public Cell(int inp_dimension) {
			dimension = inp_dimension;
			corner = new double [dimension];
			width = new double [dimension];
		}
		
		public Cell(int inp_dimension, double [] inp_corner, double [] inp_width) {
			dimension = inp_dimension;
			corner = new double [dimension];
			width  = new double [dimension];
			for(int d = 0; d < dimension; d++) setCorner(d, inp_corner[d]);
			for(int d = 0; d < dimension; d++) setWidth( d,  inp_width[d]);
		}

		public double getCorner(int d) {
			return corner[d];
		}

		public double getWidth(int d) {
			return width[d];
		}

		public void setCorner(int d, double val) {
			corner[d] = val;
		}

		public void setWidth(int d, double val) {
			width[d] = val;
		}

		// Checks whether a point lies in a cell
		public boolean containsPoint(double point[])
		{
			for(int d = 0; d < dimension; d++) {
				if(corner[d] - width[d] > point[d]) return false;
				if(corner[d] + width[d] < point[d]) return false;
			}
			return true;
		}

	}

	public class CenterMass{
		private List<Double> negf;
		private Double sumQ;
		
		public CenterMass(List<Double> negf, Double sumQ) {
			this.negf = negf;
			this.sumQ = sumQ;
		}

		public void setNegf(List<Double> negf) {
			this.negf = negf;
		}
		public void setSumQ(Double sumQ) {
			this.sumQ = sumQ;
		}
		public List<Double> getNegf() {
			return negf;
		}
		public Double getSumQ() {
			return sumQ;
		}
	}


// Default constructor for SPTree -- build tree, too!
	public SPTree(int D, double [] inp_data, int N)
	{

		// Compute mean, width, and height of current map (boundaries of SPTree)
		int nD = 0;
		double [] mean_Y = new double [D];
		double []  min_Y = new double [D]; for(int d = 0; d < D; d++)  min_Y[d] =  Double.MAX_VALUE;
		double []  max_Y = new double [D]; for(int d = 0; d < D; d++)  max_Y[d] = -Double.MAX_VALUE;
		for(int n = 0; n < N; n++) {
			for(int d = 0; d < D; d++) {
				mean_Y[d] += inp_data[n * D + d];
				if(inp_data[nD + d] < min_Y[d]) min_Y[d] = inp_data[nD + d];
				if(inp_data[nD + d] > max_Y[d]) max_Y[d] = inp_data[nD + d];
			}
			nD += D;
		}
		for(int d = 0; d < D; d++) mean_Y[d] /= (double) N;

		// Construct SPTree
		double [] width = new double [D];
		for(int d = 0; d < D; d++) width[d] = Math.max(max_Y[d] - mean_Y[d], mean_Y[d] - min_Y[d]) + 1e-5;
		init(null, D, inp_data, mean_Y, width);
		fill(N);

		// Clean up memory
	}


// Constructor for SPTree with particular size and parent -- build the tree, too!
	public SPTree(int D, double [] inp_data, int N, double [] inp_corner, double [] inp_width)
	{
		init(null, D, inp_data, inp_corner, inp_width);
		fill(N);
	}


// Constructor for SPTree with particular size (do not fill the tree)
	public SPTree(int D, double [] inp_data, double [] inp_corner, double [] inp_width)
	{
		init(null, D, inp_data, inp_corner, inp_width);
	}


// Constructor for SPTree with particular size and parent (do not fill tree)
	public SPTree(SPTree inp_parent, int D, double [] inp_data, double [] inp_corner, double [] inp_width) {
		init(inp_parent, D, inp_data, inp_corner, inp_width);
	}


// Constructor for SPTree with particular size and parent -- build the tree, too!
	public SPTree(SPTree inp_parent, int D, double [] inp_data, int N, double [] inp_corner, double [] inp_width)
	{
		init(inp_parent, D, inp_data, inp_corner, inp_width);
		fill(N);
	}

	private SPTree parent;
	private int dimension;
	private int no_children;
	private SPTree[] children;
	private double [] data;
	private Boolean is_leaf;
	private int size;
	private int cum_size;
	private Cell boundary;
	private double [] center_of_mass;
	private double [] buff;
	private int [] index;

	// Fixed constants
	static int QT_NODE_CAPACITY = 1;


	// Main initialization function
	public void init(SPTree inp_parent, int D, double [] inp_data, double [] inp_corner, double [] inp_width)
	{
		parent = inp_parent;
		dimension = D;
		no_children = 2;
		for(int d = 1; d < D; d++) no_children *= 2;
		data = inp_data;
		is_leaf = true;
		size = 0;
		cum_size = 0;
		
		index = new int[QT_NODE_CAPACITY];

		boundary = new Cell(dimension);
		for(int d = 0; d < D; d++) boundary.setCorner(d, inp_corner[d]);
		for(int d = 0; d < D; d++) boundary.setWidth(d, inp_width[d]);

		children = new SPTree [no_children];
		for(int i = 0; i < no_children; i++) children[i] = null;

		center_of_mass = new double [D];
		for(int d = 0; d < D; d++) center_of_mass[d] = .0;

		buff = new double [D];
	}


	// Update the data underlying this tree
	public void setData(double [] inp_data)
	{
		data = inp_data;
	}


// Get the parent of the current tree
	public SPTree getParent()
	{
		return parent;
	}
	


	// Insert a point into the SPTree
	public boolean insert(int new_index)
	{
		// Ignore objects which do not belong in this quad tree
		double [] point = new double [dimension];//data + new_index * dimension;
		for (int d = 0; d < dimension; d++) {
			point[d] = data[dimension*new_index + d];
		}
		
		if(!boundary.containsPoint(point))
			return false;

		// Online update of cumulative size and center-of-mass
		cum_size++;
		double mult1 = (double) (cum_size - 1) / (double) cum_size;
		double mult2 = 1.0 / (double) cum_size;
		for(int d = 0; d < dimension; d++) center_of_mass[d] *= mult1;
		for(int d = 0; d < dimension; d++) center_of_mass[d] += mult2 * point[d];

		// If there is space in this quad tree and it is a leaf, add the object here
		if(is_leaf && size < QT_NODE_CAPACITY) {
			index[size] = new_index;
			size++;
			return true;
		}

		// Don't add duplicates for now (this is not very nice)
		boolean any_duplicate = false;
		for(int n = 0; n < size; n++) {
		boolean duplicate = true;
		for(int d = 0; d < dimension; d++) {
			if(point[d] != data[index[n] * dimension + d]) { duplicate = false; break; }
		}
		any_duplicate = any_duplicate | duplicate;
	}
		if(any_duplicate) return true;

		// Otherwise, we need to subdivide the current cell
		if(is_leaf) subdivide();

		// Find out where the point can be inserted
		for(int i = 0; i < no_children; i++) {
		if(children[i].insert(new_index)) return true;
	}

		// Otherwise, the point cannot be inserted (this should never happen)
		return false;
	}


	// Create four children which fully divide this cell into four quads of equal area
	public void subdivide() {

		// Create new children
		double [] new_corner = new double[dimension];
		double [] new_width  = new double[dimension];
		for(int i = 0; i < no_children; i++) {
			int div = 1;
			for(int d = 0; d < dimension; d++) {
				new_width[d] = .5 * boundary.getWidth(d);
				if((i / div) % 2 == 1) new_corner[d] = boundary.getCorner(d) - .5 * boundary.getWidth(d);
				else                   new_corner[d] = boundary.getCorner(d) + .5 * boundary.getWidth(d);
				div *= 2;
			}
			children[i] = new SPTree(this, dimension, data, new_corner, new_width);
		}

		// Move existing points to correct children
		for(int i = 0; i < size; i++) {
			boolean success = false;
			for(int j = 0; j < no_children; j++) {
				if(!success) success = children[j].insert(index[i]);
			}
			index[i] = -1;
		}

		// Empty parent node
		size = 0;
		is_leaf = false;
	}


	// Build SPTree on dataset
	public void fill(int N)
	{
		for(int i = 0; i < N; i++) insert(i);
	}


	
	// Checks whether the specified tree is correct
	public boolean isCorrect()
	{
		for(int n = 0; n < size; n++) {
			double [] point = new double [dimension];//data + index[n] * dimension;
			for (int d = 0; d < dimension; d++) {
				point[d] = data[index[n] * dimension + d];
			}
			if(!boundary.containsPoint(point)) return false;
		}
		if(!is_leaf) {
			boolean correct = true;
			for(int i = 0; i < no_children; i++) correct = correct && children[i].isCorrect();
			return correct;
		}
		else return true;
	}



	// Build a list of all indices in SPTree
	public void getAllIndices(int [] indices)
	{
		getAllIndices(indices, 0);
	}

	public int getAllIndices(int [] indices, int loc)
	{

		// Gather indices in current quadrant
		for(int i = 0; i < size; i++) indices[loc + i] = index[i];
		loc += size;

		// Gather indices in children
		if(!is_leaf) {
			for(int i = 0; i < no_children; i++) loc = children[i].getAllIndices(indices, loc);
		}
		return loc;
	}


	public int getDepth() {
		if(is_leaf) return 1;
		int depth = 0;
		for(int i = 0; i < no_children; i++) depth = Math.max(depth, children[i].getDepth());
		return 1 + depth;
	}


	// Compute non-edge forces using Barnes-Hut algorithm
	public CenterMass computeNonEdgeForces(int point_index, double theta)
	{

		// Make sure that we spend no time on empty nodes or self-interactions
		if(cum_size == 0 || (is_leaf && size == 1 && index[0] == point_index)) return null;

		// Compute distance between point and center-of-mass
		double D = .0;
		int ind = point_index * dimension;
		for(int d = 0; d < dimension; d++) buff[d] = data[ind + d] - center_of_mass[d];
		for(int d = 0; d < dimension; d++) D += buff[d] * buff[d];

		// Check whether we can use this node as a "summary"
		double max_width = 0.0;
		double cur_width;
		for(int d = 0; d < dimension; d++) {
			cur_width = boundary.getWidth(d);
			max_width = (max_width > cur_width) ? max_width : cur_width;
		}
		if(is_leaf || max_width / Math.sqrt(D) < theta) {
			
			List<Double> neg_f = new ArrayList<Double>();
			Double sum_Q = 0.0;

			// Compute and add t-SNE force between point and current node
			D = 1.0 / (1.0 + D);
			double mult = cum_size * D;
			sum_Q += mult;
			mult *= D;
			for (int d = 0; d < dimension; d++) {
				neg_f.add(d,mult * buff[d]);
				
				//System.out.println(point_index + " index ->" + (mult * buff[d]) + "sumQ: " + sum_Q);
			}
			return new CenterMass(neg_f, sum_Q);
		}
		else {

			// Recursively apply Barnes-Hut to children
			for(int i = 0; i < no_children; i++) {
				CenterMass result = children[i].computeNonEdgeForces(point_index, theta);
				if (result != null) {
					return result;
				}
			}
		}
		return null;
	}


	// Print out tree
	public void print(int tabs)
	{

		for(int i = 0; i < tabs; i++) {
			System.out.print("\t");
		}

		if(cum_size == 0) {
			System.out.print("Empty node\n");
			return;
		}

		if(is_leaf) {
			System.out.print("Leaf node; data = [");
			for(int i = 0; i < size; i++) {
				double [] point = new double [dimension];//data + index[n] * dimension;
				for (int d = 0; d < dimension; d++) {
					point[d] = data[index[i] * dimension + d];
				}
				
				
				for(int d = 0; d < dimension; d++) System.out.print(point[d] + ",");
				System.out.print(" (index = " + index[i]);
				if(i < size - 1) System.out.print("\n");
				else System.out.print("]\n");
			}
		}
		else {
			System.out.print("Intersection node with center-of-mass = [");
			for(int d = 0; d < dimension; d++) System.out.print(center_of_mass[d] + ", ");
			System.out.print("]; children are:\n");
			for(int i = 0; i < no_children; i++) {
				System.out.print(i + " ");
				children[i].print(tabs + 1);
			}
		}
	}


}
