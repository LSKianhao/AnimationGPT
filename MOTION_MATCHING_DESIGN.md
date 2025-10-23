# Motion Matching System Design for AnimationGPT

**Inspired by:** Autodesk MotionMaker, Unreal Engine Motion Matching

**Goal:** Build an AI-powered motion library that learns transitions between poses using the CMP combat animation dataset (8,700 animations).

---

## 🎯 Overview

Instead of generating animations from text, we'll build a **motion database** that:
1. Indexes all 8,700 combat animations by pose features
2. Allows querying: "I'm in Pose A, I want to reach Pose B"
3. Returns the best matching motion fragment and smooth transitions
4. Works in real-time for game engines (Unreal/Unity)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    OFFLINE PHASE                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  CMP Dataset (8,700 animations)                        │
│         ↓                                               │
│  Motion Segmentation                                    │
│  - Break into clips (1-3 seconds)                      │
│  - Extract 50k+ motion fragments                       │
│         ↓                                               │
│  Feature Extraction                                     │
│  - Pose (joint positions/velocities)                   │
│  - Trajectory (future path)                            │
│  - Foot contacts                                        │
│  - Combat-specific (weapon pose, attack phase)         │
│         ↓                                               │
│  Database Building                                      │
│  - Build k-d tree / FAISS index                        │
│  - Store motion fragments with metadata                │
│  - Precompute transition costs                         │
│                                                         │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    ONLINE PHASE (Real-time)             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Query Input:                                           │
│  - Current Pose A                                       │
│  - Desired Pose B / Trajectory                         │
│  - Constraints (foot contacts, etc.)                   │
│         ↓                                               │
│  Search Database                                        │
│  - k-NN search in feature space                        │
│  - Find best matching fragments (< 1ms)                │
│         ↓                                               │
│  Transition Synthesis                                   │
│  - Blend between current and target                    │
│  - Smooth interpolation                                 │
│  - Maintain foot contacts                              │
│         ↓                                               │
│  Output: Animation Clip                                 │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 Feature Representation

For each frame in the database, extract:

### 1. Pose Features (22 joints × 3D)
```python
pose_features = [
    joint_positions,      # (22, 3) - current positions
    joint_velocities,     # (22, 3) - movement direction
    bone_orientations,    # (21, 4) - quaternions
    root_transform        # (7,) - position + rotation
]
```

### 2. Trajectory Features
```python
trajectory_features = [
    future_positions,     # (5, 3) - next 0.1s, 0.2s, 0.3s, 0.4s, 0.5s
    future_directions,    # (5, 2) - XZ plane directions
    future_velocities     # (5,) - speeds
]
```

### 3. Combat-Specific Features
```python
combat_features = [
    weapon_position,      # (3,) - weapon tip location
    attack_phase,         # (1,) - windup/strike/recovery
    stance,              # (1,) - neutral/aggressive/defensive
    balance_metric       # (1,) - weight distribution
]
```

### 4. Contact Features
```python
contact_features = [
    foot_contacts,        # (4,) - binary for each foot joint
    contact_normals,      # (4, 3) - ground normal at contact
    contact_velocities    # (4,) - sliding detection
]
```

**Total Feature Dimension:** ~200 features per frame

---

## 🔍 Similarity Search

### Distance Metric

```python
def motion_distance(query_features, candidate_features):
    """Compute weighted distance between motion features"""
    
    # Pose distance (high weight)
    pose_dist = weighted_euclidean(
        query_features.pose, 
        candidate_features.pose,
        weights=[2.0, 1.5, 1.0, ...]  # Joint importance
    )
    
    # Trajectory distance (medium weight)
    traj_dist = euclidean(
        query_features.trajectory,
        candidate_features.trajectory
    )
    
    # Contact distance (high weight - important for quality)
    contact_dist = binary_distance(
        query_features.contacts,
        candidate_features.contacts
    )
    
    # Combat feature distance
    combat_dist = euclidean(
        query_features.combat,
        candidate_features.combat
    )
    
    # Weighted sum
    total_dist = (
        0.4 * pose_dist +
        0.3 * traj_dist +
        0.2 * contact_dist +
        0.1 * combat_dist
    )
    
    return total_dist
```

### Search Algorithm

**Option 1: k-d Tree** (Fast for low dimensions)
- Build k-d tree on reduced features (PCA to ~50D)
- Query time: O(log N)
- Good for exact k-NN

**Option 2: FAISS** (Scalable for large databases)
- Facebook AI Similarity Search
- Supports GPU acceleration
- Query time: < 1ms for 50k fragments
- Better for approximate k-NN

**Option 3: Learned Embedding** (AI-powered)
- Train neural network to embed motions
- Similar motions → close in embedding space
- More flexible, learns what makes good transitions

---

## 🎬 Motion Database Structure

```python
class MotionDatabase:
    """
    Database of motion fragments with fast search.
    """
    
    def __init__(self):
        self.fragments = []  # List of MotionFragment objects
        self.index = None    # FAISS/k-d tree index
        
    def add_animation(self, animation_data, metadata):
        """Break animation into fragments and add to database"""
        # Segment into 1-3 second clips
        fragments = self.segment_animation(animation_data)
        
        for fragment in fragments:
            # Extract features
            features = self.extract_features(fragment)
            
            # Store fragment
            self.fragments.append(MotionFragment(
                data=fragment,
                features=features,
                metadata=metadata,
                source_animation_id=animation_id
            ))
    
    def build_index(self):
        """Build search index after adding all animations"""
        # Extract all feature vectors
        feature_matrix = np.array([f.features for f in self.fragments])
        
        # Build FAISS index
        self.index = faiss.IndexFlatL2(feature_dim)
        self.index.add(feature_matrix)
    
    def query(self, current_pose, desired_target, k=10):
        """
        Find best matching fragments.
        
        Args:
            current_pose: Current character pose
            desired_target: Target pose or trajectory
            k: Number of candidates to return
            
        Returns:
            List of (fragment_id, distance) tuples
        """
        # Build query feature vector
        query_features = self.build_query_features(
            current_pose, 
            desired_target
        )
        
        # Search index
        distances, indices = self.index.search(
            query_features.reshape(1, -1), 
            k
        )
        
        # Return best matches
        return [
            (self.fragments[idx], dist) 
            for idx, dist in zip(indices[0], distances[0])
        ]
```

---

## 🔄 Transition Synthesis

Once we find the best matching fragment, blend it smoothly:

### 1. Inertial Blending
```python
def blend_transition(current_clip, target_clip, blend_frames=10):
    """
    Smooth transition between two clips.
    
    Uses inertialization to avoid pops.
    """
    # Compute velocity at transition point
    current_vel = current_clip[-1] - current_clip[-2]
    target_vel = target_clip[1] - target_clip[0]
    
    # Create blending curve (ease-in-ease-out)
    blend_curve = ease_in_out_curve(blend_frames)
    
    # Blend positions and velocities
    blended = []
    for i, alpha in enumerate(blend_curve):
        # Position blend
        pos = lerp(current_clip[-1], target_clip[i], alpha)
        
        # Velocity blend (smoother)
        vel = slerp(current_vel, target_vel, alpha)
        
        # Combine
        blended.append(pos + vel * dt)
    
    return blended
```

### 2. Foot Contact Preservation
```python
def preserve_foot_contacts(transition, contact_labels):
    """Ensure feet don't slide during transition"""
    
    for foot_idx in [3, 4, 7, 8]:  # Ankle and toe joints
        if contact_labels[foot_idx]:
            # Foot should be planted
            planted_position = transition[0][foot_idx]
            
            # Lock position during contact
            for frame in transition:
                frame[foot_idx] = planted_position
```

---

## 📦 Dataset Processing Pipeline

### Step 1: Load CMP Dataset
```python
# CMP has 8,700 animations in HumanML3D format
# new_joints/ contains (T, 22, 3) joint positions
# texts/ contains annotations (not used for matching)

dataset_path = "datasets/humanml3d/"
animations = load_all_animations(dataset_path)
# Result: 8,700 animations
```

### Step 2: Segment into Fragments
```python
# Break each animation into overlapping clips
# - Clip length: 30-90 frames (1-3 seconds @ 30fps)
# - Overlap: 15 frames (0.5 seconds)
# - Result: ~50,000 motion fragments

fragments = []
for anim in animations:
    clips = sliding_window(anim, window=60, stride=15)
    fragments.extend(clips)

# Total: ~50,000 fragments
```

### Step 3: Feature Extraction
```python
# For each fragment, compute features
database = MotionDatabase()

for fragment in fragments:
    features = extract_all_features(fragment)
    database.add_fragment(fragment, features)

database.build_index()
# Database ready for queries!
```

---

## 🎮 Query Interface

### Use Case 1: Pose-to-Pose Transition

**Problem:** Character is in idle stance, wants to do a sword attack

```python
# Current state
current_pose = get_character_pose()  # (22, 3)

# Desired target
target_pose = create_target_pose(
    weapon="katana",
    action="overhead_attack",
    direction=[0, 0, 1]  # Forward
)

# Query database
results = database.query(
    current_pose=current_pose,
    desired_target=target_pose,
    k=5  # Get top 5 matches
)

# Get best match
best_fragment, distance = results[0]

# Generate transition
transition = synthesize_transition(
    current_pose,
    best_fragment,
    blend_frames=10
)

# Play animation
play(transition)
```

### Use Case 2: Trajectory Following

**Problem:** Character running, needs to turn left while maintaining combat stance

```python
# Current state
current_pose = get_character_pose()
current_velocity = get_velocity()

# Desired trajectory (next 0.5 seconds)
desired_trajectory = [
    current_position + velocity * dt
    for dt in [0.1, 0.2, 0.3, 0.4, 0.5]
]

# Add turn (left 45 degrees over 0.5s)
for i, point in enumerate(desired_trajectory):
    angle = -45 * (i / len(desired_trajectory))
    point = rotate(point, angle)

# Query
results = database.query_trajectory(
    current_pose=current_pose,
    trajectory=desired_trajectory,
    constraints={"maintain_combat_stance": True}
)

# Synthesize motion
motion = synthesize_from_matches(results, desired_trajectory)
play(motion)
```

### Use Case 3: Action Blending

**Problem:** Character is attacking, player presses dodge button

```python
# Current state
current_animation = "sword_swing"
current_frame = 15  # Mid-swing

# Desired action
target_action = "dodge_roll"
target_direction = "left"

# Query with high priority on responsiveness
results = database.query_action_interrupt(
    current_state=(current_animation, current_frame),
    target_action=target_action,
    direction=target_direction,
    priority="responsive"  # vs "smooth"
)

# Create interruption transition
transition = create_interrupt_blend(
    current=current_animation[current_frame:],
    target=results[0],
    method="quick"  # 5 frames vs 10 frames
)

play(transition)
```

---

## 🚀 Performance Optimization

### Indexing Strategy
- **Total fragments:** ~50,000
- **Feature dimension:** 200 → reduce to 64 with PCA
- **Index type:** FAISS IVF (Inverted File Index)
- **Query time:** < 1ms on CPU, < 0.1ms on GPU

### Caching
```python
class MotionMatchingRuntime:
    def __init__(self, database):
        self.database = database
        self.cache = LRUCache(maxsize=1000)
    
    def query(self, current_pose, target):
        # Check cache first
        cache_key = hash_query(current_pose, target)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Query database
        results = self.database.query(current_pose, target)
        
        # Cache results
        self.cache[cache_key] = results
        return results
```

### Precomputation
- Precompute transition costs between common poses
- Build transition graph for frequently used actions
- Store blending weights for character-specific tweaks

---

## 🎨 Advantages for Game Development

### 1. **Real-time Performance**
- Query: < 1ms
- Transition synthesis: < 5ms
- Total: < 10ms (60fps = 16.6ms budget) ✅

### 2. **Artist Control**
- Artists can tag animations with metadata
- Fine-tune feature weights
- Manually specify "good" transitions

### 3. **Runtime Adaptation**
- No retraining needed
- Add new animations on-the-fly
- Character-specific databases (heavy vs light character)

### 4. **Quality Guarantee**
- Uses real captured data (no generation artifacts)
- Foot sliding eliminated (contact-aware search)
- Natural motion (from real combat animations)

### 5. **Integration**
- Export to Unreal Motion Matching
- Export to Unity Kinematica
- Custom runtime for other engines

---

## 📈 Comparison: Text-to-Motion vs Motion Matching

| Aspect | Text-to-Motion | Motion Matching |
|--------|---------------|-----------------|
| **Use Case** | Offline content creation | Real-time gameplay |
| **Quality** | Variable (generation artifacts) | Perfect (real data) |
| **Control** | Vague text prompts | Precise pose/trajectory |
| **Speed** | Slow (seconds) | Real-time (< 1ms) |
| **Dataset Use** | Training data | Direct database |
| **Memory** | Model weights (100MB-1GB) | Motion database (50-200MB) |
| **GPU Required** | Yes (inference) | No (CPU search fine) |
| **Integration** | Hard | Native to engines |
| **New Animations** | Retrain model | Add to database |

**Verdict:** Motion Matching is **far superior** for game engines! 🎮

---

## 🛠️ Implementation Plan

### Phase 1: Database Builder (Week 1)
- [ ] Load CMP dataset (8,700 animations)
- [ ] Segment into fragments (~50k clips)
- [ ] Extract basic features (pose, velocity, trajectory)
- [ ] Build FAISS index
- [ ] Save database to disk

### Phase 2: Query System (Week 1-2)
- [ ] Implement query interface (Pose A → Pose B)
- [ ] Implement k-NN search
- [ ] Add feature weighting
- [ ] Test query speed (target < 1ms)

### Phase 3: Transition Synthesis (Week 2)
- [ ] Implement blending algorithms
- [ ] Add foot contact preservation
- [ ] Implement inertialization
- [ ] Quality testing

### Phase 4: Combat Features (Week 2-3)
- [ ] Add weapon-specific features
- [ ] Add attack phase detection
- [ ] Add stance classification
- [ ] Combat-aware search

### Phase 5: Runtime & Export (Week 3-4)
- [ ] Optimize for real-time (caching, precomputation)
- [ ] Export to Unreal Motion Matching format
- [ ] Export to Unity Kinematica
- [ ] Create demo application

### Phase 6: Advanced Features (Week 4+)
- [ ] Learned embeddings (neural network)
- [ ] Automatic transition graph building
- [ ] Multi-character support
- [ ] Online learning (add animations at runtime)

---

## 📚 References

### Similar Systems
- **Unreal Engine Motion Matching:** https://dev.epicgames.com/documentation/en-us/unreal-engine/motion-matching-in-unreal-engine
- **Autodesk MotionMaker:** https://blogs.autodesk.com/media-and-entertainment/2025/06/04/meet-motionmaker/
- **Unity Kinematica:** https://unity.com/products/kinematica
- **Ubisoft LaForge (Learned Motion Matching):** https://montreal.ubisoft.com/en/introducing-learned-motion-matching/

### Academic Papers
- "Motion Matching and The Road to Next-Gen Animation" (Buttner & Clavet, GDC 2015)
- "Learned Motion Matching" (Holden et al., SIGGRAPH 2020)
- "Neural State Machine for Character-Scene Interactions" (Starke et al., SIGGRAPH Asia 2019)

### Datasets
- **CMP (CombatMotionProcessed):** 8,700 combat animations (we have this!)
- **CMR (CombatMotionRaw):** 14,883 animations (can add later)

---

## 🎯 Success Criteria

The motion matching system will be successful if:

1. **Query speed:** < 1ms for k-NN search (50k fragments)
2. **Quality:** No visible pops or artifacts in transitions
3. **Coverage:** Can transition between any two poses in < 0.5s
4. **Integration:** Works in Unreal/Unity
5. **Scalability:** Can handle 100k+ fragments with < 10ms queries

---

**Next Steps:** Implement Phase 1 (Database Builder) 🚀
