
# 🧠 **Operating System - Unit 5: I/O and File Management**

---

## 📘 1. **Organization of I/O Functions**

I/O (Input/Output) operations are essential for any computer system to interact with the outside world. The Operating System (OS) handles complex communication between user programs and physical hardware through a layered architecture.

### 🔹 Goals of I/O System:
- Efficient and secure data transmission between memory and devices.
- Standard interface for different devices.
- Coordination between multiple I/O devices.
  
### 🔹 Layers of I/O System:
1. **User-Level I/O:**
   - High-level programming functions used in user programs.
   - Examples: `scanf()`, `printf()`, `fread()`, `fwrite()` in C.
   - Easy for programmers but requires OS support.

2. **Device-Independent I/O:**
   - Provides uniform interfaces for devices regardless of hardware.
   - Responsibilities:
     - Naming devices (e.g., `/dev/usb0`)
     - Buffering, error handling, protection

3. **Device Drivers:**
   - Specific programs that understand hardware communication.
   - Translate high-level commands into device-specific commands.
   - Loaded during system boot or dynamically.

4. **Interrupt Handlers:**
   - Devices generate **interrupts** when they complete operations.
   - The OS has handlers to respond and resume suspended processes.

> **Example:** When you copy a file to a pen drive, OS invokes user-level copy command, uses buffering, communicates with the USB driver, and manages interrupts when the copy is done.

---

## 📘 2. **I/O Buffering**

I/O devices are often slower than the CPU, so buffering is used to match speeds and improve performance.

### 🔹 Buffer:
A **buffer** is a portion of memory set aside to store data temporarily while it is being moved from one place to another.

### 🔹 Types of Buffering:

1. **Single Buffering:**
   - Only one buffer is used.
   - CPU has to wait if data is not yet ready.

2. **Double Buffering:**
   - Two buffers are used in parallel.
   - While one buffer is being used, the other is being filled.
   - Improves efficiency, reduces CPU wait time.

3. **Circular Buffering:**
   - A ring of multiple buffers.
   - Data flows in a circular manner.
   - Commonly used in real-time and multimedia applications.

> **Example:** When you stream a YouTube video, it preloads upcoming data using circular buffering to avoid lags.

---



# 📘 3 **Disk Scheduling in Operating System**

---

## 💡 What is Disk Scheduling?

When multiple processes request data from a disk (like a hard drive), the **Operating System** needs to decide **which request to serve first**. This process is called **Disk Scheduling**.

### 🎯 **Why Disk Scheduling is Important?**
- Reduces **seek time** (time to move disk arm to required track)
- Improves system **performance** and **response time**
- Manages **multiple requests efficiently**

---

## 🔄 Components of Disk Access Time

1. **Seek Time**: Time to move disk arm to desired track  
2. **Rotational Latency**: Time for the desired sector to rotate under read/write head  
3. **Transfer Time**: Time to actually read/write data

> ⚠️ Out of these, **Seek Time** is the most variable and hence optimization target.

---

## 📊 Disk Scheduling Algorithms

### 1. 🟢 **FCFS (First Come First Serve)**
- Requests are processed **in the order** they arrive.
- **Simple** to implement, but not always efficient.

#### ✅ Example:
Requests: 98, 183, 37, 122  
Initial Head at: 100  
Servicing Order: 98 → 183 → 37 → 122  
🕒 Seek Time = Total arm movement = (100→98)+(98→183)+(183→37)+(37→122) = **236 cylinders**

#### 🔴 Cons:
- High average seek time
- May serve far-away requests first

---

### 2. 🟢 **SSTF (Shortest Seek Time First)**
- Chooses request **closest to current head** position.
- Reduces total movement.

#### ✅ Example:
Requests: 98, 183, 37, 122  
Initial Head at: 100  
Closest = 98 → then 122 → then 183 → then 37  
🕒 Total Movement = (100→98)+(98→122)+(122→183)+(183→37) = **208 cylinders**

#### ⚠️ Drawback:
- Can cause **starvation** of far requests

---

### 3. 🟢 **SCAN (Elevator Algorithm)**
- Disk arm moves **in one direction**, serving all requests until end, then reverses.
- Like an **elevator**: goes up, then down.

#### ✅ Example:
Head at: 50  
Requests: 10, 20, 35, 70, 90  
Moves: 35 → 20 → 10 → then turns → 70 → 90

#### 🔄 Advantage:
- Fair to all processes  
- Better average seek time than FCFS

---

### 4. 🟢 **LOOK**
- Like SCAN but **only goes as far as the last request** in each direction, not till disk end.

#### 🔄 Advantage:
- Less head movement than SCAN

---

### 5. 🟢 **C-SCAN (Circular SCAN)**
- Head moves in **one direction** only.
- After reaching the end, it jumps back to the beginning **without serving requests** on the way back.

#### ✅ Example:
Requests: 20, 40, 50, 130, 150  
Head moves: 50 → 130 → 150 → (jump back to 20) → 40

#### ✅ Advantage:
- **Uniform wait time** for all requests

---

### 6. 🟢 **C-LOOK**
- Same as C-SCAN, but only goes as far as **last request**, not end of disk.

#### 🔄 Example:
Requests: 20, 30, 100, 120  
Head: 50  
Movement: 100 → 120 → jump to 20 → 30

---

## 📌 Comparison Table

| Algorithm | Type | Pros | Cons |
|-----------|------|------|------|
| FCFS | Non-Preemptive | Simple | High wait time |
| SSTF | Non-Preemptive | Fast | Starvation |
| SCAN | Elevator | Fair | Some delay |
| LOOK | Optimized SCAN | Less movement | Slightly complex |
| C-SCAN | Circular | Uniform wait | More movement |
| C-LOOK | Circular + Optimized | Fast & Fair | Moderate complexity |

---

## 🧠 Summary

- **Disk Scheduling** = Optimize which request is served first
- **Goal** = Minimize **seek time**
- Use **SSTF** for performance, **SCAN/C-SCAN** for fairness
- Real-world systems use combinations or improvements of these

---


## 📘 4. **RAID (Redundant Array of Independent Disks)**

RAID is a data storage technique using multiple disks to improve **performance** and **fault tolerance**.

### 🔹 Benefits:
- Increased reliability
- Improved read/write speed
- Data redundancy

### 🔹 Common RAID Levels:

1. **RAID 0 – Striping:**
   - Splits data across multiple disks.
   - No redundancy. If one fails, data is lost.
   - High speed.

2. **RAID 1 – Mirroring:**
   - Duplicates data on two or more disks.
   - Provides fault tolerance.
   - Slower write speed, but safe.

3. **RAID 5 – Striping with Parity:**
   - Data is striped across disks with parity bits.
   - If one disk fails, data can be recovered using parity.
   - Balanced performance and reliability.

> **Example:** RAID 1 is used in servers where data safety is crucial.

---

## 📘 5. **File Concept**

A **file** is a named collection of related information stored on secondary storage (like HDD, SSD, or USB).

### 🔹 File Attributes:
- **Name**: File identifier (e.g., notes.txt)
- **Type**: Executable, text, image, etc.
- **Location**: Directory path
- **Size**: In bytes
- **Time, Date**: Creation, last modified
- **Permissions**: Read, write, execute

### 🔹 File Operations:
- **Create** – Create a new file.
- **Open** – Load file into memory.
- **Read/Write** – Transfer data to/from file.
- **Reposition** – Move file pointer.
- **Close** – Finish working on file.
- **Delete** – Remove file from storage.

> **Example:** When you open a `.docx` file, OS reads its metadata (name, type, permissions) before loading it.

---

## 📘 6. **Access Methods**

Different ways to access or retrieve file data depending on the type of application.

### 🔹 Types:

1. **Sequential Access:**
   - Read data in order, one by one.
   - Cannot skip; slow for large files.
   - Used in log files, audio/video.

2. **Direct (Random) Access:**
   - Jump to any location in file using index.
   - Faster, flexible.
   - Used in databases.

3. **Indexed Access:**
   - Uses an index or map to locate data blocks.
   - Combines speed of direct access with flexibility of sequential access.

> **Example:** Playing an audio file from the middle requires direct access.

---

## 📘 7. **Directory Structures**

A directory is a container for files and other directories. It helps in organizing files efficiently.

### 🔹 Types of Directory Structures:

1. **Single-Level Directory:**
   - All files in one folder.
   - Simple, but causes name conflicts.

2. **Two-Level Directory:**
   - Separate directory for each user.
   - Solves name conflict but still limited.

3. **Tree-Structured Directory:**
   - Hierarchical structure (folders, subfolders).
   - Used in Windows/Linux systems.

4. **Acyclic Graph Directory:**
   - Allows sharing files among users.
   - Prevents cycles (no infinite loops).

5. **General Graph Directory:**
   - Similar to acyclic, but allows cycles.
   - Requires garbage collection and reference counting.

> **Example:** C:\Users\Abhishek\Documents\Project is part of a tree-structured directory.

---

## 📘 8. **Protection**

Protection ensures that only authorized users can access or modify files and system resources.

### 🔹 Types of Protection:

1. **Access Control:**
   - Each file has a list specifying who can read/write/execute.
   - Example: Admin can modify, others can read only.

2. **Password Protection:**
   - Some files are accessible only after entering a password.

3. **User Groups and Permissions:**
   - Group similar users and assign permissions.
   - Example: In Linux, `chmod`, `chown` commands.

### 🔹 Protection in OS:
- Prevents malware, data leakage, accidental deletion.
- Ensures multi-user environment is secure.

> **Example:** In college labs, students can only access their folders, not others'.
