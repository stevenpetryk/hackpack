<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/9.9.0/highlight.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlightjs-line-numbers.js/1.1.0/highlightjs-line-numbers.min.js"></script>
<script>hljs.initHighlightingOnLoad();</script>
<script>hljs.initLineNumbersOnLoad();</script>
<link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Roboto|Source+Code+Pro" />
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/9.9.0/styles/github.min.css" />
<link rel="stylesheet" href="main.css" />
<meta charset="utf-8" />

<!--
```java
import java.util.*;
```
-->

# General Programming

## Comparable

Sorting from smallest to largest:

<!--
```java
class Thing {
  int value;
``` -->

```java
// a.compareTo(b) < 0   =>   a < b   =>   a - b < 0
public int compareTo (Thing other) {
  return value - other.value;
}
```

<!--
```java
}
``` -->

<div class="page-break"></div>

## Geometry classes

```java
// Code modified from Arup Guha's geometry routines
// Found at http://www.cs.ucf.edu/~dmarino/progcontests/cop4516/samplecode/Test2DGeo.java

class Point {
  public double x, y, z;

  public Point(double _x, double _y) {
    this(_x, _y, 0);
  }

  public Point(double _x, double _y, double _z) {
    x = _x;
    y = _y;
    z = _z;
  }

  public boolean isStraightLineTo (Point mid, Point end) {
    Vector from = new Vector(this, mid);
    Vector to = new Vector(mid, end);

    return from.isStraightLineTo(to);
  }

  public boolean isRightTurn(Point mid, Point end) {
    Vector from = new Vector(this, mid);
    Vector to = new Vector(mid, end);

    return from.isLeftTurnTo(to);
  }

  public Vector getVector(Point to) {
    return new Vector(to.x - x, to.y - y, to.z - z);
  }

  public String toString () {
    return "<" + x + ", " + y + ">";
  }
}
```

<div class="page-break"></div>

```java
class Vector {
	public double x, y, z;

  public Vector(double _x, double _y) {
    this(_x, _y, 0);
  }

	public Vector(double _x, double _y, double _z) {
		x = _x;
		y = _y;
		z = _z;
	}

  public Vector (Point start, Point end) {
    x = end.x - start.x;
    y = end.y - start.y;
  }

  public double dot (Vector other) {
    return this.x * other.x + this.y * other.y + this.z * other.z;
  }

	public Vector crossProduct(Vector other) {
		return new Vector((y * other.z) - (other.y * z), (z * other.x) - (other.z * x), (x * other.y) - (other.x * y));
	}

	public double magnitude() {
		return Math.sqrt((x * x) + (y * y) + (z * z));
	}

  public double angle(Vector other) {
    return Math.acos(this.dot(other) / magnitude() / other.magnitude());
  }

  public double signedCrossMag(Vector other) {
    return this.x * other.y - other.x * this.y;
  }

  public double crossProductMagnitude (Vector other) {
    return Math.abs(signedCrossMag(other));
  }

  public double referenceAngle () {
    return Math.atan2(y, x);
  }

  public boolean isStraightLineTo (Vector other) {
    return signedCrossMag(other) == 0;
  }

  public boolean isLeftTurnTo (Vector other) {
    return signedCrossMag(other) > 0;
  }
}
```

<div class="page-break"></div>

```java
class Line {
  final public static double EPSILON = 1e-9;

  public Point p, end;
  public Vector dir;

  public Line(Point _start, Point _end) {
    p = _start;
    end = _end;
    dir = new Vector(p, end);
  }

  public Point intersect(Line other) {
    double den = det(dir.x, -other.dir.x, dir.y, -other.dir.y);
    if (Math.abs(den) < EPSILON) return null;

    double numLambda = det(other.p.x-p.x, -other.dir.x, other.p.y-p.y, -other.dir.y);
    return eval(numLambda/den);
  }

  public Point getPoint(double t) {
    return new Point(p.x + dir.x * t, p.y + dir.y * t, p.z + dir.z * t);
  }

  public double distance(Point other) {
    Vector toPt = new Vector(p, other);
    return dir.crossProductMagnitude(toPt) / dir.magnitude();
  }

  public Point eval(double lambda) {
    return new Point(p.x + lambda * dir.x, p.y + lambda * dir.y);
  }

  public static double det(double a, double b, double c, double d) {
    return a * d - b * c;
  }
}

class Plane {
	public Point a, b, c;
	public Vector normalVector;
	public double distanceToOrigin;

	public Plane(Point _a, Point _b, Point _c) {
		a = _a;
		b = _b;
		c = _c;
		Vector v1 = a.getVector(b);
		Vector v2 = a.getVector(c);
		normalVector = v1.crossProduct(v2);
		distanceToOrigin = (normalVector.x * a.x) + (normalVector.y * a.y) + (normalVector.z * a.z);
	}

	public boolean onPlane(Point p) {
		return (normalVector.x * p.x) + (normalVector.y * p.y) + (normalVector.z * p.z) == distanceToOrigin;
	}
}
```

<div class="page-break"></div>

# Permutation Generation

Example: print out all alphabetic strings of a given length.

```java
class WordInventor {
  static List<String> results;

  public static List<String> generateCombinations (int length) {
    results = new ArrayList<String>();
    generateCombinations(length, "", 0);
    return results;
  }

  public static void generateCombinations (int length, String accumulator, int k) {
    if (k == length) {
      results.add(accumulator);
      return;
    }

    for (char c = 'a'; c <= 'z'; c++) {
      generateCombinations(length, accumulator + c, k + 1);
    }
  }
}
```

<div class="page-break"></div>

# Combination Generation
<div class="page-break"></div>

<!--
```java
class MathUtils {
```
-->

# GCD

```java
public static int gcd (int a, int b) {
  return b == 0 ? a : gcd(b, a%b);
}
```

# LCM

```java
public static int lcm (int a, int b) {
  return a * (b / gcd(a, b));
}
```

<div class="page-break"></div>

<!--
```java
}
```
-->

# Graphs

```java
class Node {
  int value;
  public List<Edge<Node>> children;
  public Node () { this(0); }
  public Node (int _value) { value = _value; children = new ArrayList<Edge<Node>>(); }

  public Node addChild (Node child, int weight) {
    return addChild(child, weight, true);
  }

  public Node addChild (Node child, int weight, boolean reciprocate) {
    children.add(new Edge<>(this, child, weight));
    if (reciprocate) child.addChild(this, weight, false); // if undirected graph

    return this;
  }
}

class Edge<T> implements Comparable<Edge> {
  T node, from; int weight;
  Edge (T _node, int _weight) { this(null, _node, _weight); }
  Edge (T _from, T _node, int _weight) { from = _from; node = _node; weight = _weight; }

  @Override
  public int compareTo (Edge other) {
    return weight - other.weight;
  }
}
```

<div class="page-break"></div>

## Kruskal's Algorithm

```java
class Kruskal {
  public static int getMSTWeight (Node start, int numNodes) {
    Queue<Edge<Node>> edges = new PriorityQueue<>();
    edges.add(new Edge<Node>(null, start, 0));

    int result = 0;

    DisjointSet ds = new DisjointSet(5);

    int nodesReached = 0;

    while (!edges.isEmpty()) {
      Edge<Node> currentEdge = edges.poll();
      Node currentNode = currentEdge.node;

      boolean merged = true;
      if (currentEdge.from != null) {
        merged = ds.union(currentEdge.from.value, currentEdge.node.value);
      }

      if (!merged) continue;
      nodesReached++;
      edges.addAll(currentNode.children);
      result += currentEdge.weight;
    }

    return nodesReached == numNodes ? result : -1;
  }
}

class DisjointSet {
  int[] parent, rank;

  public DisjointSet (int n) {
    rank = new int[n]; parent = new int[n];

    for (int i = 0; i < n; i++) parent[i] = i;
  }

  public int find (int value) {
    if (parent[value] != value) parent[value] = find(parent[value]);
    return parent[value];
  }

  public boolean union (int a, int b) {
    int aRoot = find(a);
    int bRoot = find(b);
    if (aRoot == bRoot) return false;

    if      (rank[aRoot] < rank[bRoot]) parent[aRoot] = bRoot;
    else if (rank[aRoot] > rank[bRoot]) parent[bRoot] = aRoot;
    else {
      parent[bRoot] = aRoot;
      rank[aRoot]++;
    }

    return true;
  }
}
```

<div class="page-break"></div>

## Prim's Algorithm

```java
class Prim {
  public static int getMSTWeight (Node start, int numNodes) {
    Queue<Edge<Node>> pq = new PriorityQueue<>();
    Set<Node> visited = new HashSet<>();

    int result = 0;

    pq.add(new Edge<Node>(start, 0));

    while (!pq.isEmpty()) {
      Edge<Node> current = pq.poll();
      Node currentNode = current.node;
      if (!visited.add(currentNode)) continue;

      result += current.weight;

      pq.addAll(currentNode.children);
    }

    if (visited.size() == numNodes) {
      return result;
    } else {
      return -1;
    }
  }
}
```

<div class="page-break"></div>

## Depth First Search
```java
class DFS {
  public static boolean canReachNode (Node start, Node target) {
    Set<Node> visited = new HashSet<>();
    Deque<Node> queue = new ArrayDeque<>();
    queue.push(start);

    while (!queue.isEmpty()) {
      Node current = queue.pop();

      if (!visited.add(current)) continue;
      if (current == target) return true;

      for (Edge<Node> edge : current.children) {
        queue.push(edge.node);
      }
    }

    return false;
  }
}
```
<div class="page-break"></div>

## Breadth First Search

```java
class BFS {
  public static int distanceToNode (Node start, Node target) {
    Set<Node> visited = new HashSet<>();
    Deque<NodeWithDistance> queue = new ArrayDeque<>();
    queue.add(new NodeWithDistance(start, 0));

    while (!queue.isEmpty()) {
      NodeWithDistance current = queue.poll();

      if (!visited.add(current.node)) continue;
      if (current.node == target) return current.distance;

      for (Edge<Node> edge : current.node.children) {
        queue.add(new NodeWithDistance(edge.node, current.distance + 1));
      }
    }

    return -1;
  }

  static class NodeWithDistance {
    Node node; int distance;
    public NodeWithDistance (Node _node, int _distance) { node = _node; distance = _distance; }
  }
}
```

<div class="page-break"></div>

# Topological Sort
<div class="page-break"></div>

# Floyd-Warshall's Algorithm

```java

class FloydWarshalls {
  public static int[][] floydwarshalls(int[][] matrix) {
    int n = matrix.length;
    int[][] sp = new int[n][n];

    for (int i = 0; i < n; i++)
			for (int j = 0; j < n; j++)
				sp[i][j] = (i == j) ? 0 : matrix[i][j];

    // Floyd-Warshall's    
    for (int k = 1; k <= n; k++)
			for (int i = 0; i < n; i++)
				for (int j = 0; j < n; j++)
					sp[i][j] = Math.min(sp[i][j], sp[i][k-1] + sp[k-1][j]);

    // Negative cycle detection.
    for (int i = 0; i < n; i++)
      if (sp[i][i] < 0)
        return new int[1][1];

    return sp;
  }
}

```

<div class="page-break"></div>

# Dijkstra's Algorithm
```java
class Dijkstras {
  public static LinkedList<Vertex> dijkstras(int source, int[][] matrix) {
    int[] dist = new int[matrix.length];
    boolean[] visited = new boolean[matrix.length];
    int numVisited = 0;
    PriorityQueue<Vertex> queue = new PriorityQueue<>();
    LinkedList<Vertex> path = new LinkedList<>();

    Arrays.fill(dist, Integer.MAX_VALUE);
    dist[source] = 0;


    for(int i = 0; i < matrix.length; i++)
      queue.add(new Vertex(i, dist[i]));

    while (!queue.isEmpty() && numVisited < matrix.length) {
      Vertex vertex = queue.remove();
      if(visited[vertex.id]) continue;
      visited[vertex.id] = true;
      path.add(vertex);

      for(int i = 0; i < matrix.length; i++) {
        if(matrix[vertex.id][i] > 0 && !visited[i] && dist[vertex.id] + matrix[vertex.id][i] < dist[i]) {
          dist[i] = dist[vertex.id] + matrix[vertex.id][i];
					queue.add(new Vertex(i, dist[i]));
        }
      }
    }

    return path;
  }

  static class Vertex {
    int id; int distance;
    public Vertex (int _id, int _distance) {
      id = _id; distance = _distance;
    }
  }
}
```
<div class="page-break"></div>

# Bellman Ford's Algorithm
<div class="page-break"></div>

# Network Flow
<div class="page-break"></div>

# Matrix Chain Multiplication
<div class="page-break"></div>

# Longest Common Subsequence
```java
class LCS {
  public static int longestCommonSubsequenceLength (String x, String y) {
    int lengths[][] = new int[x.length() + 1][y.length() + 1];

    Arrays.fill(lengths[0], 0);
    for (int i = 0; i < lengths.length; i++) lengths[i][0] = 0;

    for (int i = 1; i < lengths.length; i++) {
      for (int j = 1; j < lengths[0].length; j++) {
        if (x.charAt(i - 1) == y.charAt(j - 1)) {
          lengths[i][j] = lengths[i - 1][j - 1] + 1;
        } else {
          lengths[i][j] = Math.max(lengths[i - 1][j], lengths[i][j - 1]);
        }
      }
    }

    return lengths[lengths.length - 1][lengths[0].length - 1];
  }
}
```

<div class="page-break"></div>

# Knapsack DP

```java
class Knapsack {
  public static int knapsack (int capacity, int[] weights, int[] values, boolean allowDups) {
    int n = weights.length;
    int[] dp = new int[capacity + 1];

    for (int i = 0; i < n; i++) {
      for (
        int w = allowDups ? weights[i] : capacity;
        allowDups ? w <= capacity : w >= weights[i];
      ) {
        dp[w] = Math.max(dp[w], dp[w-weights[i]] + values[i] );

        if (allowDups) w++; else w--;
      }
    }

    return dp[capacity];
  }
}
```

<div class="page-break"></div>

# Line-Line Intersection

```java
class LineLineIntersection {

  public static Point intersection(Line line1, Line line2) {
    return line1.intersect(line2);
  }
}
```

<div class="page-break"></div>

# Line-Plane Intersection

```java
class LinePlaneIntersection {
  final public static double EPSILON = 1e-9;

  public static Point intersection(Plane p, Line l) {

    double t = (p.normalVector.x * l.dir.x) +
               (p.normalVector.y * l.dir.y) +
               (p.normalVector.z * l.dir.z);

    if(Math.abs(t) < EPSILON)
      return null;

    double parameter = p.distanceToOrigin -
                       (p.normalVector.x * l.p.x) -
                       (p.normalVector.y * l.p.y) -
                       (p.normalVector.z * l.p.z);

    return l.getPoint(parameter / t);
  }                                  
}

```
<div class="page-break"></div>

# Polygon Area
<div class="page-break"></div>

# Convex Hull

```java
class ConvexHullSolver {
  int numPoints;
  Queue<Point> initialPoints;
  Queue<Point> sortedPoints;
  Point firstPoint;

  public static Comparator<Point> getLowerLeftComparator() {
    return new Comparator<Point>() {
      @Override
      public int compare(Point o1, Point o2) {
        if (o1.y != o2.y) return Double.compare(o1.y, o2.y);

        return Double.compare(o1.x, o2.x);
      }
    };
  }

  public static Comparator<Point> getReferenceAngleComparator (final Point initialPoint) {
    return new Comparator<Point>() {
      @Override
      public int compare(Point p1, Point p2) {
        if (p1 == initialPoint) return -1;
        if (p2 == initialPoint) return 1;

        Vector v1 = new Vector(initialPoint, p1);
        Vector v2 = new Vector(initialPoint, p2);

        if (Math.abs(v1.referenceAngle() - v2.referenceAngle()) < 1e-4) {
          return Double.compare(v1.magnitude(), v2.magnitude());
        }

        return Double.compare(v1.referenceAngle(), v2.referenceAngle());
      }
    };
  }

  public ConvexHullSolver (int _numPoints) {
    numPoints = _numPoints;
    initialPoints = new PriorityQueue<>(numPoints, getLowerLeftComparator());
  }

  public void addPoint (Point point) {
    initialPoints.add(point);
  }
```

<div class="page-break"></div>

```java
  public Stack<Point> solve () {
    sortPoints();

    Stack<Point> pointStack = new Stack<>();

    if (sortedPoints.size() <= 3) {
      List<Point> points = new ArrayList<>(sortedPoints);

      if (points.get(0).isStraightLineTo(points.get(1), points.get(2))) {
        pointStack.add(points.get(0));
        pointStack.add(points.get(1));
      } else {
        pointStack.addAll(sortedPoints);
      }

      return pointStack;
    }

    pointStack.push(sortedPoints.poll());
    pointStack.push(sortedPoints.poll());

    while (!sortedPoints.isEmpty()) {
      Point endPoint = sortedPoints.poll();
      Point midPoint = pointStack.pop();
      Point prevPoint = pointStack.pop();

      while (!prevPoint.isRightTurn(midPoint, endPoint)) {
        if (pointStack.isEmpty()) {
          midPoint = endPoint;
          endPoint = sortedPoints.poll();
        } else {
          midPoint = prevPoint;
          prevPoint = pointStack.pop();
        }
      }

      pointStack.push(prevPoint);
      pointStack.push(midPoint);
      pointStack.push(endPoint);
    }

    return pointStack;
  }

  public void sortPoints () {
    firstPoint = initialPoints.peek();

    sortedPoints = new PriorityQueue<>(numPoints, getReferenceAngleComparator(firstPoint));
    sortedPoints.addAll(initialPoints);
  }
}
```

<div class="page-break"></div>

# Point in Polygon
<div class="page-break"></div>

# Tests

```java
public class hackpack {
  public static boolean failures = false;

  public static void main (String args[]) {
    testCombinationGeneration();
    testGCD();
    testLCM();
    testDisjointSet();
    testKruskals();
    testPrims();
    testDFS();
    testBFS();
    testFloydWarshalls();
    testDijkstras();
    testLCS();
    testKnapsack();
    testConvexHull();

    if (!failures) {
      handleSuccess();
    }
  }

  public static void testCombinationGeneration () {
    List<String> results = WordInventor.generateCombinations(3);
    assertEqual(results.size(), (int)Math.pow(26, 3));
  }

  public static void testGCD () {
    assertEqual(MathUtils.gcd(1, 1), 1);
    assertEqual(MathUtils.gcd(5, 10), 5);
    assertEqual(MathUtils.gcd(15, 3), 3);
  }

  public static void testLCM () {
    assertEqual(MathUtils.lcm(1, 1), 1);
    assertEqual(MathUtils.lcm(5, 10), 10);
    assertEqual(MathUtils.lcm(8, 3), 24);
  }

  public static void testDisjointSet () {
    DisjointSet set = new DisjointSet(5);

    set.union(1, 2);
    set.union(1, 3);
    assertEqual(set.find(2), 1);
    assertEqual(set.find(3), 1);
    assertEqual(set.find(4), 4);
  }

  public static void testKruskals () {
    Node a = new Node(0), b = new Node(1), c = new Node(2), d = new Node(3), e = new Node(4);

    a.addChild(b, 1);
    a.addChild(c, 2);
    c.addChild(e, 3);
    e.addChild(a, 4);

    assertEqual(Kruskal.getMSTWeight(a, 5), -1);

    e.addChild(d, 5);
    assertEqual(Kruskal.getMSTWeight(a, 5), 11);
  }

  public static void testPrims () {
    Node a = new Node(0), b = new Node(1), c = new Node(2), d = new Node(3), e = new Node(4);

    a.addChild(b, 1);
    a.addChild(c, 2);
    c.addChild(e, 3);
    e.addChild(a, 4);

    assertEqual(Prim.getMSTWeight(a, 5), -1);

    e.addChild(d, 5);
    assertEqual(Prim.getMSTWeight(a, 5), 11);
  }

  public static void testDFS () {
    Node start = new Node();
    Node reachable = new Node();
    Node unreachable = new Node();

    start
      .addChild(new Node(), 1)
      .addChild(new Node(), 1)
      .addChild(
        new Node()
          .addChild(reachable, 1)
          .addChild(new Node(), 1),
        1
      );

    assertTrue(DFS.canReachNode(start, reachable), "Expected node to be reachable");
    refute(DFS.canReachNode(start, unreachable), "Expected node to be unreachable");
  }

  public static void testBFS () {
    Node start = new Node();
    Node reachable = new Node();
    Node unreachable = new Node();

    start
      .addChild(new Node(), 1)
      .addChild(new Node(), 1)
      .addChild(
        new Node()
          .addChild(reachable, 1)
          .addChild(new Node(), 1),
        1
      );

    assertEqual(BFS.distanceToNode(start, reachable), 2);
    assertEqual(BFS.distanceToNode(start, unreachable), -1);
  }

  public static void testFloydWarshalls() {
    // int[][] matrix = new int[4][4];
    // int[][] result = new int[4][4];
    //FloydWarshalls fw = new FloydWarshalls(4, matrix);

    int[][] matrix = {{1000000000, 1000000000, -2, 1000000000},
              {4, 1000000000, 3, 1000000000},
              {1000000000, 1000000000, 1000000000, 2}, {1000000000, -1, 1000000000, 1000000000}};
    int[][] result = {{0, -1, -2, 0},
              {4, 0, 2, 4},
              {5, 1, 0, 2},
              {3, -1, 1, 0}};

    assertArraysEqual(FloydWarshalls.floydwarshalls(matrix), result);
  }

  public static void testDijkstras() {
    int[][] matrix = new int[9][9];
  }

  public static void testLCS () {
    String x = "123456789";
    String y = "13597341234569";
                    // ^^^^^^^

    assertEqual(LCS.longestCommonSubsequenceLength(x, y), 7);
  }

  public static void testKnapsack () {
    int weights[] = new int[] { 3, 2, 6, 8, 1, 3 };
    int values[] = new int[] { 7, 5, 12, 20, 3, 6 };

    assertEqual(Knapsack.knapsack(1, weights, values, false), 3);
    assertEqual(Knapsack.knapsack(2, weights, values, false), 5);
    assertEqual(Knapsack.knapsack(10, weights, values, false), 25);
    assertEqual(Knapsack.knapsack(23, weights, values, false), 53);
  }

  public static void testConvexHull () {
    ConvexHullSolver solver = new ConvexHullSolver(5);
    List<Point> points = new ArrayList<>();

    Point topLeft   = new Point(2, 0), topRight   = new Point(2, 2),
            lowerLeft = new Point(0, 0), lowerRight = new Point(0, 2),
            middle    = new Point(1, 1);

    solver.addPoint(lowerLeft);
    solver.addPoint(lowerRight);
    solver.addPoint(topLeft);
    solver.addPoint(topRight);
    solver.addPoint(middle);

    Stack<Point> hull = solver.solve();

    assertContains(hull, lowerLeft);
    assertContains(hull, lowerRight);
    assertContains(hull, topLeft);
    assertContains(hull, topRight);
    refuteContains(hull, middle);
  }

  /*
   * Low-level test code. Don't worry about this too much.
   */

  public static final String ANSI_RESET = "\u001B[0m";
  public static final String ANSI_RED = "\u001B[31m";
  public static final String ANSI_GREEN = "\u001B[32m";

  private static void handleSuccess () {
    System.out.println(ANSI_GREEN + "✓ All tests passed" + ANSI_RESET);
  }

  private static void handleTestFailure (TestFailure e) {
    failures = true;

    String failingTest = "";

    outer: for (StackTraceElement element : e.getStackTrace()) {
      String name = element.getMethodName();

      if (!name.startsWith("assert") && !name.startsWith("throwOn")) {
        failingTest = name;
        break outer;
      }
    }

    System.out.println(ANSI_RED + "× " + failingTest + " failed: " + e.getMessage() + ANSI_RESET);

    e.printStackTrace();
  }

  private static <T> void assertEqual (T a, T b) {
    assertTrue(a.equals(b), String.format("Expected %s to equal %s", a, b));
  }

  private static <T> void assertArraysEqual (T[] a, T[] b) {
    assertTrue(
      Arrays.deepEquals(a, b),
      String.format("Expected %s to match %s", Arrays.deepToString(a), Arrays.deepToString(b))
    );
  }

  private static <T> void assertContains (List<T> haystack, T needle) {
    assertTrue(haystack.contains(needle), String.format("Expected %s to contain %s", haystack, needle));
  }

  private static <T> void refuteContains (List<T> haystack, T needle) {
    refute(haystack.contains(needle), String.format("Expected %s not to contain %s", haystack, needle));
  }

  private static void assertTrue (boolean thing, String message) {
    try {
      if (!thing) {
        throw new TestFailure(message);
      }
    } catch (TestFailure e) {
      handleTestFailure(e);
    }
  }

  private static void refute (boolean thing, String message) {
    assertTrue(!thing, message);
  }
}

class TestFailure extends Exception {
  public TestFailure (String message) {
    super(message);
  }
}
```
