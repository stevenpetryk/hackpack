<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/9.9.0/highlight.min.js"></script>
<script>hljs.initHighlightingOnLoad();</script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/9.9.0/styles/github.min.css" />

# Problem Solving Hack Pack

## General Programming

### Comparable

Sorting from smallest to largest:

```java
// a.compareTo(b) < 0   =>   a < b   =>   a - b < 0
public int compareTo (Thing other) {
  return value - other.value;
}
```

## Greedy Algorithms

### Room scheduling

Given a single room to schedule, and a list of requests, the goal of this problem is to
maximize the total number of events scheduled. Each request simply consists of the
group, a start time and an end time during the day

```java
public class Main {
  public static void main (String args[]) {
    Queue<Room> pq = new PriorityQueue();

    // (load queue)

    Deque<Room> scheduledRooms = new ArrayDeque();
    while (!pq.isEmpty()) {
      Room current = pq.poll();
      if (scheduledRooms.size() > 0 && current.start < scheduledRooms.getLast().end) continue;

      scheduledRooms.add(current);
    }
  }
}

class Room implements Comparable<Room> {
  String name;
  int start, end;

  public Room (String name, int start, int end) {
    this.name = name; this.start = start; this.end = end;
  }

  public int compareTo (Room other) {
    return end - other.end;
  }
}
```

To modify this for multiple rooms, just spill over into the next room when the current room can't be scheduled.
