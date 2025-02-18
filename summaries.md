# Guided Conditional Diffusion for Controllable Traffic Simulation

**Authors:** Ziyuan Zhong, Davis Rempe, Danfei Xu, Yuxiao Chen, Sushant Veer, Tong Che, Baishakhi Ray, Marco Pavone

**Year:** 2022

This paper introduces a novel approach called Controllable Traffic Generation (CTG) that bridges the gap between realism and controllability in traffic simulation—a key requirement for developing and validating autonomous vehicles (AVs). The central idea is to generate realistic, human-like trajectories while allowing users to specify high-level objectives (such as reaching a target speed, following a waypoint, or avoiding collisions) at test time. The approach leverages the strengths of conditional diffusion models and formal, differentiable specifications of rules grounded in signal temporal logic (STL).

Key details and contributions include:

1. Problem Statement and Motivation  
 • Traffic simulation for AV testing must combine real-world realistic behavior with the ability to enforce user-specified objectives (e.g., obeying speed limits, avoiding collisions).  
 • Traditional heuristic or learning-based simulators either offer controllability with simple rule-based systems or realism by reproducing dataset statistics, but rarely both.  
 • CTG formulates the problem of controllable traffic generation where the goal is to produce future trajectories that balance natural driving patterns with imposed rules.

2. Methodology: Conditional Diffusion with Guidance  
 • CTG uses a conditional diffusion model that is trained offline on real-world driving data from the nuScenes dataset. The model learns to generate trajectories by predicting actions, with states obtained by propagating these actions through a known vehicle dynamics model (a unicycle model). This ensures that the generated trajectories are not only realistic but also physically feasible.  
 • During inference, the model is guided by incorporating differentiable objectives defined using STL. The STL framework provides a syntax to specify complex spatio-temporal rules (such as “vehicles must not collide” or “must follow a target speed”) along with a robustness measure that quantifies how well a trajectory satisfies these rules.
 • Rather than relying on pre-trained classifiers or reward networks, CTG directly computes gradients from the robustness functions of STL formulas. These gradients are then used to perturb the denoising steps of the diffusion process, thereby steering the generated trajectory toward rule-compliant behaviors.
 • The guidance process is extended to multi-agent settings to handle interaction-based rules (e.g., collision avoidance), with an iterative joint optimization procedure that updates all agents simultaneously.

3. Experimental Evaluation  
 • The paper evaluates CTG on the nuScenes dataset, benchmarking its performance against strong baselines—such as SimNet, TrafﬁcSim, and BITS (and their variants augmented with test-time optimization).  
 • Metrics include rule violation (using STL-based evaluation metrics for various rules like speed limit, target speed, collision avoidance, off-road behavior, and reaching a waypoint), realism deviation (computed via the Wasserstein distance of driving statistical profiles), and failure rate (such as collisions or road departures).
 • Results show that CTG achieves notably lower rule violations and maintains high realism, striking a favorable balance between the two. It performs particularly well in multi-rule scenarios (e.g., combining stop sign compliance with off-road avoidance or simultaneously following waypoints and target speeds).

4. Design Choices and Ablation Studies  
 • An extensive ablation study reveals the contributions of various components: enforcing the vehicle dynamics through state supervision, the use of guidance (STL-based gradients), sample filtration, and additional inner-loop optimization steps.  
 • Each component is demonstrated to be essential for achieving the low rule violations while preserving natural driving behavior; for instance, without guidance the generated trajectories frequently violate the specified rules, and without enforcing dynamics overall quality degrades.

5. Significance and Future Work  
 • The work establishes a framework that promotes controllable behavior synthesis by integrating diffusion models with differentiable logic-based guidance.  
 • Although the current focus is on vehicle simulation, the approach is general enough to be extended to other agents such as cyclists or pedestrians.  
 • The authors suggest that long-term, robust simulation of interactive traffic scenarios and incorporating dynamic re-planning for AV testing are promising future directions.

In summary, the paper presents a state-of-the-art diffusion-based framework that successfully generates realistic traffic behaviors compliant with user-specified STL rules. This offers an effective tool for simulation-driven development and safety verification of autonomous vehicles, merging the benefits of both data-driven realism and high-level controllability.

---

# Model-Based Diffusion for Trajectory Optimization

**Authors:** Chaoyi Pan, Zeji Yi, Guanya Shi, Guannan Qu

**Year:** 2024

This paper introduces Model-Based Diffusion (MBD), a novel framework that reformulates trajectory optimization as a sampling problem guided by model information through a diffusion process. The authors leverage explicit knowledge of the system dynamics (and constraints) to compute the score function directly, rather than relying solely on demonstration data as in traditional diffusion models. This approach enables efficient, data-free generation of dynamically feasible trajectories, while still allowing for natural integration with imperfect or partial demonstration data when available.

Key aspects of the work include:

1. Theoretical Formulation and Algorithmic Design:
 • The paper recasts trajectory optimization as sampling from a target distribution defined by the cost function, dynamics (feasibility) constraints, and additional constraints. Unlike model‐free diffusion frameworks that learn the score function from data, MBD computes the gradient of the log probability (score function) by leveraging the known model information.
 • A Monte Carlo score ascent method is proposed to iteratively refine samples from an isotropic Gaussian noise distribution. In each backward diffusion step, the current noisy sample is updated by a gradient ascent step on the log probability density, scaled appropriately by a scheduling parameter derived from the forward process. This procedure progressively transforms an easy-to-sample Gaussian into an optimal trajectory that minimizes the cost.

2. Extension to Constrained Trajectory Optimization:
 • For practical trajectory optimization, the target distribution must account not only for optimality (minimizing a cost function) but also for dynamic feasibility and other constraints. The paper shows how to incorporate these by using a Dirac delta to enforce the dynamics and by explicitly incorporating constraint functions into the score estimation.
 • The authors address potential issues in sampling efficiency for feasible trajectories, particularly for long-horizon tasks, by “rolling out” samples through the dynamics (similar to a shooting method) and re-weighting them according to both the cost and constraint satisfaction.

3. Integration with Demonstration Data:
 • Recognizing that high-quality demo data may not always be available, the framework is designed to incorporate demonstrations of varying quality (e.g., partial-state, dynamically infeasible trajectories) as auxiliary observations. A balancing constant is adaptively determined so that when the model’s likelihood is low, the method relies on the demonstration; otherwise, it prioritizes the model-based solution.
 • This demonstration augmentation helps steer the diffusion process toward desired, natural, or robust solutions, particularly for complex tasks like humanoid jogging where human motion data can regularize the outcome.

4. Empirical Evaluation and Connections to Other Optimization Methods:
 • MBD is rigorously evaluated on various challenging contact-rich and high-dimensional control tasks, including locomotion (e.g., hopper, half-cheetah, ant, humanoid tasks) and manipulation (e.g., pushT). The experiments indicate that MBD not only outperforms state-of-the-art reinforcement learning algorithms (e.g., PPO, SAC) but also traditional sampling-based optimization methods (e.g., CMA-ES, CEM, MPPI) in terms of both performance and computation time.
 • The paper also demonstrates that when the diffusion process is reduced to a single step, MBD resembles the Cross-Entropy Method (CEM), highlighting theoretical connections between diffusion-based strategies and classical sampling-based optimizers.
 • Beyond control tasks, the method is shown to be effective in high-dimensional black-box optimization benchmarks (such as the Ackley and Rastrigin functions) and even in gradient-free training of deep neural networks for MNIST classification, illustrating broad applicability.

Overall, this work offers a new perspective on trajectory optimization by embedding the learning process within a diffusion framework that is explicitly informed by system dynamics. The ability to combine a model-based approach with demonstration augmentation enables MBD to generate high-quality, feasible trajectories in complex, non-convex, and non-smooth environments while maintaining computational efficiency. Future directions mentioned include deeper theoretical analysis of convergence, adaptation to online control through receding horizon strategies, and further enhancements in sampling and scheduling.

---

# Scaling Robot Learning with Semantically Imagined Experience

**Authors:** Tianhe Yu, Ted Xiao, Austin Stone, Jonathan Tompson, Anthony Brohan, Su Wang, Jaspiar Singh, Clayton Tan, Dee M, Jodilyn Peralta, Brian Ichter, Karol Hausman, Fei Xia

**Year:** 2023

This paper introduces ROSIE (Robot Learning with Semantically Imagened Experience), a novel framework that leverages state-of-the-art text-to-image diffusion models to augment robotic manipulation datasets. The core idea is to synthetically generate semantically meaningful variations of real-world robot demonstrations in order to scale up the amount and diversity of training data without requiring additional real-world data collection or labor-intensive demonstrations.

Key components and insights include:

1. Purpose and Motivation  
 • The paper addresses the challenge of acquiring diverse, high-scale robot data necessary for robust manipulation skills. Traditional approaches either rely on human demonstrations or complex autonomous data collection, which are both difficult and expensive to scale.  
 • ROSIE provides a complementary solution by using foundation models from computer vision and NLP (e.g., diffusion models and large language models) to modify existing datasets semantically. This enables robots to learn new skills and become more robust to unknown objects, backgrounds, and distractors.

2. Methodology  
 • ROSIE operates by augmenting video episodes from a robotic dataset. For each episode, it first detects the relevant regions in each image using open-vocabulary segmentation (employing models like OWL-ViT with an additional mask head). This segmentation isolates areas where new objects or distractors can be inserted while preserving critical task-related regions.  
 • The system then generates augmentation prompts. Two approaches are presented:
  – Hand-engineered prompts allow designers to directly specify changes.  
  – LLM-proposed prompts, which use a large language model such as GPT-3 to generate detailed, semantically rich augmentation text.  
 • With these textual prompts and the localized masks, a text-guided inpainting model (Imagen Editor) is used to edit the images. This process replaces or adds objects (e.g., substituting the contents of a drawer, altering a table background, or replacing a chip bag with a microfiber cloth) in a photorealistic manner while keeping the rest of the scene unmodified.  
 • The augmented episodes retain the original action sequences and are paired with modified task instructions reflecting the changes. The mixture of original and augmented data is then used to fine-tune a Robotics Transformer (RT-1) policy.

3. Experimental Evaluation and Results  
 • The authors test ROSIE in a range of robotic manipulation tasks that progressively become more challenging:
  – Learning completely new manipulation skills by inserting unseen target objects (e.g., replacing a chip bag with various cloth types) while preserving grasp and action consistency.
  – Robustifying manipulation policies by changing object placement contexts such as novel containers (e.g., turning a top drawer into a metal sink) or backgrounds (e.g., replacing a table top with colorful table cloths), thus challenging the policy to operate in previously unseen visual environments.
  – Enhancing high-level embodied reasoning tasks such as success detection. By augmenting manipulation episodes with distractors and clutter (for example, adding extra objects in a drawer), the paper demonstrates that a CLIP-based success detector can achieve significantly improved performance on out-of-distribution (OOD) cases.
 • Across several comparative experiments—including a baseline pre-trained RT-1 policy and a variant with only instruction (text) augmentation—the ROSIE method yields substantial performance improvements in success rate and robustness over diverse tasks.

4. Key Contributions and Discussion  
 • The paper demonstrates that off-the-shelf image generation tools can effectively scale robot learning by synthesizing rich, semantically diverse data.  
 • The semantically guided augmentations not only enable the learning of entirely new tasks (that the robot has never physically experienced) but also improve the policy’s tolerance to visual clutter and distractors in complex real-world scenes.  
 • The proposed method is computationally intensive and currently applied per frame (which may reduce temporal consistency), and it focuses primarily on augmenting visual appearances rather than motion dynamics. These aspects are noted as directions for future work, potentially by combining with simulation data or exploring more efficient generative architectures.

In summary, ROSIE outlines a promising approach to robot data augmentation by “imagining” new experiences through text-guided diffusion models. This method significantly reduces the dependency on real-world data collection while enhancing the robustness and generalization of robot manipulation policies, ultimately helping robots adapt to novel tasks and dynamic environments.

---

# Taming Transformers for Realistic Lidar Point Cloud Generation

**Authors:** Hamed Haghighi, Amir Samadi, Mehrdad Dianati, Valentina Donzella, Kurt Debattista

**Year:** 2024

The paper introduces LidarGRIT, an innovative generative model tailored for realistic Lidar point cloud synthesis. The main motivation is to overcome a limitation in existing generative models—especially Diffusion Models—which, despite their stable training and iterative refinement, do not realistically reproduce Lidar raydrop noise due to their inherent denoising process.

Key Contributions and Methodology:

1. Two-Stage Generation Process:
 • LidarGRIT first transforms 3D Lidar point clouds into range image representations (using spherical projection for KITTI-360 and scan unfolding for KITTI odometry) to leverage the efficiency of image-based generative models.
 • It employs an adapted VQ-VAE architecture to encode the range images into discrete latent tokens and then separately decodes these tokens into a clean range image and a raydrop noise mask. This disentanglement is facilitated through a specialized training strategy that uses two separate loss functions:
  – A masked reconstruction loss that makes the decoder focus on reconstructing the actual range values where points exist.
  – A binary raydrop loss that explicitly guides the network to estimate the noise mask accurately.

2. Auto-Regressive Transformer for Sampling:
 • Instead of iteratively refining in the image space (like traditional diffusion models), LidarGRIT uses an auto-regressive (AR) transformer to model the interactions between the latent tokens. This iterative token sampling in latent space enables the model to capture complex dependencies and improves the overall quality of the generated point clouds.
 • The transformer is trained in an auto-regressive fashion where each token index is predicted based on its preceding tokens, ensuring that the generation process is coherent and progressively refined.

3. Geometric Preservation:
 • To counteract the overfitting encountered by VQ-VAE models when handling low-resolution range images, the authors incorporate geometric preservation strategies. By applying geometric transformations (such as affine transformations and flips) on the input images during training, the model is encouraged to preserve the underlying geometry, leading to more expressive and generalizable latent representations.

Evaluation and Insights:

• Extensive experiments on the KITTI-360 and KITTI odometry datasets demonstrate that LidarGRIT outperforms state-of-the-art models across several quantitative metrics. For instance, it shows superior results on image-based metrics like Sliced Wasserstein Distance (SWD) and on point cloud quality metrics like Frechet Point Distance (FPD), Minimum-Matching Distance (MD), and Jensen-Shannon Divergence (JSD).
• An ablation study confirms the significance of both the raydrop loss and geometric preservation techniques. The inclusion of these components significantly enhances the realism in both the range image and 3D point cloud domains.

Overall, the paper makes the following key points:
 – It presents a novel hybrid approach that combines the benefits of iterative token sampling via AR transformers with a specially adapted VQ-VAE for robust latent representation.
 – It disentangles the generation of range images from raydrop noise synthesis, thereby addressing a notable shortfall in previous models.
 – It validates its approach with rigorous experiments and ablations, highlighting improved quality in generated Lidar data, which is critical for realistic simulation in autonomous vehicle applications.

In summary, LidarGRIT is a significant advancement in data-driven Lidar simulation as it offers a more stable and realistic generative framework by bridging the gap between precise 3D shape modeling and the accurate reproduction of sensor noise, setting a new benchmark in the field.

---

# SlowFast-VGen: Slow-Fast Learning for Action-Driven Long Video Generation

**Authors:** Yining Hong, Beide Liu, Maxine Wu, Yuanhao Zhai, Kai-Wei Chang, Linjie Li, Kevin Lin, Chung-Ching Lin, Jianfeng Wang, Zhengyuan Yang, Yingnian Wu, Lijuan Wang

**Year:** 2024

The paper introduces SLOWFAST-VGEN, a novel dual-speed learning framework for action-driven long video generation that mimics the complementary learning system in the human brain. The key idea is to combine two distinct learning mechanisms: a slow learning process that builds a general “world model” using a masked conditional video diffusion model, and a fast learning component that rapidly adapts to new contexts by storing episodic memory in TEMP-LORA parameters.

Slow Learning uses a latent video diffusion model conditioned on language inputs (describing actions) and previous video chunks. This component is pre-trained on a large, diverse dataset comprising 200K videos with action annotations, covering domains such as driving, gaming, human activities, robotics, and simulations. The diffusion process is adapted via a masking mechanism so that the future frames are generated conditioned on preceding frames, ensuring that initial parts of the video serve as a fixed context.

Fast Learning, on the other hand, is introduced to address the challenge of maintaining consistency over long video sequences. By employing a temporary Low-Rank Adaptation (TEMP-LORA) module, the model updates its parameters at inference time based solely on the current context and newly generated outputs. This mechanism acts like an episodic memory, storing the sequential trajectory of the video so that distant frames remain coherent with the earlier parts. Notably, TEMP-LORA is inspired by methods originally designed for long text generation, but here it is adjusted to focus on memorizing entire video trajectories rather than immediate transitions.

The model further integrates these two learning paradigms into a slow-fast learning loop. In the inner (fast) loop, the TEMP-LORA module quickly adapts and updates memory during the generation of each video chunk. In the outer (slow) loop, information collected across multiple episodes is used to update the core slow-learning parameters, effectively consolidating long-term patterns and context. This iterative loop not only improves the windowed generation but also enhances long-horizon planning tasks.

Beyond straightforward video generation, the system is also applied to video planning. Here the model transforms task descriptions into actionable video plans by generating sequential video chunks and utilizing an inverse dynamics model to translate these visual trajectories into executable actions. Two long-horizon planning tasks are demonstrated—in robot manipulation (using RLBench) and game navigation (using Minecraft)—where the method shows significant improvements in metrics that assess positional accuracy and visual fidelity.

Experimental results indicate that SLOWFAST-VGEN outperforms several baseline methods (such as AVDC, Streaming-T2V, and others) on key video quality metrics including Fréchet Video Distance (FVD), PSNR, SSIM, LPIPS, and novel metrics like Scene Revisit Consistency (SRC). For example, the proposed method achieves an FVD of 514 compared to higher scores from the baselines and exhibits significantly fewer scene cuts, indicating improved temporal consistency. Qualitative comparisons further show that the inclusion of TEMP-LORA greatly enhances the coherence, smoothness, and long-term memory retention of generated video sequences.

The paper also draws parallels between its dual-speed learning system and the brain’s neocortex (slow learning) and hippocampus (fast learning), presenting TEMP-LORA as an analogue to local learning rules in neural systems. Extensive ablation studies and evaluations confirm that each component—both the fast learning module and the slow-fast learning loop—is critical for ensuring the quality and consistency of long video outputs.

In summary, SLOWFAST-VGEN is an innovative framework that integrates slow and fast learning paradigms to generate action-conditioned long videos with high fidelity and temporal consistency. Its ability to store episodic memory directly in a lightweight adaptation module enables it to bridge gaps in context larger than the typical model window, making it suitable not only for video generation but also for complex planning tasks that require long-term consistency. Limitations include coverage gaps in the training data and additional computational overhead during inference, which are acknowledged as directions for future improvements.

---

# Safe Offline Reinforcement Learning with Real-Time Budget Constraints

**Authors:** Qian Lin, Bo Tang, Zifan Wu, Chao Yu, Shangqin Mao, Qianlong Xie, Xingxing Wang, Dong Wang

**Year:** 2023

The paper introduces TREBI—a novel algorithm for safe offline reinforcement learning (RL) that addresses settings in which safety budgets are imposed in real time. In many safety‐critical applications (e.g., autonomous systems or ad bidding), policies must abide by dynamically determined constraints during deployment. Traditional safe RL mostly focuses on online interactions with potential budget violations during training, while many offline RL methods do not adequately guarantee safety under varying, real‐time budgets.

Key Contributions and Approach:
1. Problem Formulation:  
 • The authors formalize safe RL as a constrained Markov decision process where each trajectory must satisfy a cost (or safety) budget.  
 • They extend this formulation to account for a real-time budget constraint—meaning the policy should optimally respond to different budget levels during execution without requiring retraining for each budget value.

2. Trajectory-level Perspective via Diffusion Planning:  
 • Rather than directly optimizing a state-action policy subject to expected cost constraints, TREBI casts the problem as one of trajectory optimization.  
 • It first approximates the behavior-policy’s trajectory distribution using offline data and then “guides” the generation of safe trajectories via a diffusion model framework (specifically building on the recent Diffuser method).  
 • By modifying the reverse process in the diffusion model with a guidance term that depends on both reward and cost, TREBI is able to impose strict per-trajectory safety guarantees (with probability one) rather than merely in expectation.

3. Theoretical Analysis:  
 • The paper derives an optimal trajectory distribution under safety and distribution constraints. In particular, the authors show that the optimal solution involves reweighting the behavior trajectory distribution by an exponential factor over rewards while zeroing out trajectories that violate the cost constraint.  
 • They prove an error bound on the estimation of episodic reward and cost under the offline setting and provide performance guarantees that ensure any policy derived from the optimal distribution attains a near-optimal return while keeping costs within the given safe budget (up to errors that vanish in deterministic environments or can be controlled in probabilistic settings).

4. Algorithm and Implementation:  
 • TREBI uses a two-phase procedure. First, the offline data is used to train a diffusion model that approximates the trajectory distribution. Second, during inference, the algorithm employs guided planning—a modified reverse diffusion process where gradients with respect to both the reward and cost functions steer the trajectory generation.  
 • Notably, by conditioning on different budget levels during inference, TREBI can generate trajectories that satisfy varying real-time safety budgets without the need to retrain separate policies.

5. Empirical Evaluations:  
 • Comprehensive experiments on simulation benchmarks (including Pendulum swing-up, Reacher, several MuJoCo tasks, and tasks from Bullet-Safety-Gym) show that TREBI maintains strict safety constraint satisfaction while achieving competitive episodic rewards compared to several state-of-the-art safe offline RL baselines (such as BCQ-Lagrangian, CPQ, BCORLE(λ), and BCQ-Sauté).  
 • Additional ablation studies investigate design choices (e.g., planning horizon, control frequency, and a parameter controlling conservativeness) and confirm that TREBI is robust across different settings.  
 • A real-world advertising bidding application is also presented where advertisers have varying real-time target ROIs and budgets. In this high-stakes setting, TREBI outperforms baselines by generating fewer constraint violations and better total pay improvements, demonstrating its potential in practical deployment scenarios.

Overall, the paper’s primary contribution is to recast safe offline RL with dynamic budget constraints into a trajectory optimization problem solved via diffusion model-based planning. This trajectory-level approach allows for strict per-trajectory constraint satisfaction and efficient adaptation to varying real-time budgets—all while operating in the offline setting where raw interactions with the environment are unavailable. The extensive theoretical guarantees and empirical validations suggest that TREBI is a promising framework for deploying RL safely in applications where strict and dynamic safety constraints play a critical role.

---

# SE(3)-DiffusionFields: Learning smooth cost functions for joint grasp and motion optimization through diffusion

**Authors:** Julen Urain, Niklas Funk, Jan Peters, Georgia Chalvatzaki

**Year:** 2022

This paper introduces SE(3)-DiffusionFields (SE(3)-DiF), a framework that leverages diffusion models to learn smooth, data-driven cost functions directly on the SE(3) manifold. In robotics, planning a manipulation task such as pick-and-place requires considering multiple objectives simultaneously (e.g., grasp quality, collision avoidance, joint limits, and trajectory smoothness). Traditionally, grasp pose generation and motion planning are treated separately, which can lead to infeasible grasp samples and inefficient optimization. This work addresses that by jointly optimizing grasp selection and robot motion within a single gradient‐based framework.

Key contributions and ideas include:

1. Learning Smooth Cost Functions on SE(3):  
 • The authors adapt score-based diffusion models—which have proven effective in Euclidean spaces—to the Lie group SE(3), representing full 6DoF poses (position and orientation).  
 • Instead of learning an explicit grasp generator, they learn a scalar energy field by training a neural network via denoising score matching (DSM). This energy function represents the negative log-probability of good grasp configurations and is smooth (i.e., provides meaningful gradients everywhere), which is crucial for gradient-based optimization.

2. Architecture and Training Strategy:  
 • The proposed architecture takes as input an object’s identity, pose, and (optionally) shape information (or point clouds via a pointcloud encoder) and a candidate grasp pose in SE(3).  
 • It maps the grasp pose into a set of 3D points and, through a feature encoder and decoder, outputs a scalar energy representing grasp quality.
 • The network is trained jointly to match the object’s Signed Distance Field (SDF) values (instilling geometric reasoning) and to satisfy the DSM loss on perturbed grasp poses. Noise is added in SE(3) using an adapted Gaussian model, and the inverse diffusion process is implemented by a tailored Langevin MCMC sampling over the manifold.

3. Joint Grasp and Motion Optimization:  
 • The learned SE(3)-DiF is integrated as one term into an overall cost function that also includes other task-specific heuristic costs—such as collision avoidance and trajectory smoothness.
 • By rewriting the optimization problem as an inverse diffusion process, the framework iteratively refines a set of trajectory samples (where the final end-effector pose is evaluated with the learned cost function) using gradient descent.
 • This joint approach avoids the pitfalls of decoupled optimization (where a separately sampled grasp might be unfeasible when considering the robot’s kinematics or external constraints), leading to significantly better sample efficiency and higher success rates.

4. Experimental Evaluation:  
 • In simulation (using Nvidia Isaac Gym) and on real robotic platforms, the authors demonstrate that the SE(3)-DiF model generates more diverse, high-quality 6DoF grasp poses compared to baselines like VAE-based methods and classifier-based pose refinements.
 • When used as part of the joint grasp-and-motion optimization, the proposed method achieves higher success rates in a variety of complex scenarios (e.g., picking in clutter, handling occlusions, reorienting objects before placing them on shelves) while needing far fewer trajectory samples than decoupled methods.
 • Detailed experiments, including quantitative metrics such as success rates and Earth Mover Distance (EMD), confirm that the smooth cost function provided by the diffusion model offers superior gradient information, allowing the entire optimization process to be performed end-to-end with gradient-based methods.

Overall, the paper presents a novel integration of diffusion models with motion planning, establishing a unified framework that learns and exploits smooth cost landscapes in SE(3). By jointly optimizing grasp quality and robot trajectories via inverse diffusion, the method provides a more robust and sample-efficient solution for complex robotic manipulation tasks.

---

# Compositional Diffusion Models for Powered Descent Trajectory Generation with Flexible Constraints

**Authors:** Julia Briden, Yilun Du, Enrico M. Zucchelli, Richard Linares

**Year:** 2024

This paper introduces TrajDiffuser, a novel diffusion model–based framework that generates feasible, multi-modal 6 degree-of-freedom powered descent trajectories. The core idea is to use a compositional diffusion approach that generates complete trajectories concurrently (rather than sequentially), enabling flexible handling and composition of constraints, which together provide efficient warm starts for computationally expensive trajectory optimizations.

Key points and contributions include:

1. Motivation and Background:
 • Traditional powered descent guidance methods (both direct and indirect) often require iterative, computationally-intensive optimizations. They also struggle with initialization issues and with adapting to varied or additional constraints.
 • Recent work has used AI models (e.g., deep neural networks, transformers) to generate warm-start trajectories, but these typically generate trajectories sequentially, which can accumulate errors and lack robustness.
 • Diffusion models, known for their ability to learn multi-modal distributions and generate entire “images” (or in this case, trajectory matrices comprising states and control inputs) concurrently, offer a promising solution for achieving stable long-horizon planning.

2. The Diffusion and Energy-Based Model Framework:
 • The paper details the mathematical formulation of diffusion models. It explains the forward (noising) process that gradually adds Gaussian noise and the reverse (denoising) process used to generate samples.
 • An important insight is the reinterpretation of the diffusion model as an energy-based model (EBM). This insight allows the authors to compose multiple energy functions – representing different constraints – using operations such as product, mixture, and negation.

3. Compositional Diffusion:
 • A major technical contribution is the formulation of compositional diffusion. Instead of training a single model to handle all combinations of constraints (which would require a huge dataset), the authors train models on simpler “building block” distributions.
 • They then compose these models during inference time. For example, constraints like state-triggered conditions or drag can be incorporated by taking the product or negation of the corresponding energy functions.
 • The authors derive an upper bound on the error introduced by using reverse diffusion on these composed energy functions, giving theoretical guarantees on the approximation accuracy.

4. Application to Powered Descent Guidance:
 • The paper formulates the 6 DoF powered descent guidance problem as a minimum-final-time, non-convex optimization problem with detailed state and control constraints (e.g., mass depletion, attitude dynamics, thrust limits, and orientation constraints).
 • A successive convexification (SCvx) algorithm is used to generate a dataset of optimal trajectories. These trajectories serve as training data for TrajDiffuser.
 • Additionally, state-triggered constraints are introduced – for instance, a velocity-triggered angle-of-attack constraint that limits the spacecraft’s aerodynamic load under high dynamic pressure.
 • Trajectories are stored as matrices that encapsulate position, velocity, quaternion orientation, angular velocity, mass, and thrust over the planning horizon.

5. Model Architecture and Training:
 • The TrajDiffuser uses a U-Net architecture with time-step embeddings to condition the reverse diffusion process.
 • The training minimizes a loss function based on predicting the noise (or equivalently, gradients of an energy function) that was added during the forward process.
 • The model is trained on trajectory data generated via SCvx and then scales the outputs using a RobustScaler to handle outliers inherent in optimization outputs.
 • Cosine beta scheduling is used to control the noise variance throughout the diffusion process.

6. Experimental Validation and Results:
 • The paper compares trajectories generated by TrajDiffuser with those produced by a conventional numerical optimizer. Results show that TrajDiffuser produces trajectories that lie in similar regions of the state space and are close to locally optimal.
 • Detailed comparisons for states (position, velocity, attitude, angular velocity) and control inputs validate that the diffusion-generated trajectories respect the dynamics and constraints reasonably well.
 • Constraint violations are measured and analyzed. Although some errors are observed (notably in altitude and thrust prediction), the mean-squared-error analysis indicates that most variables remain close to the locally optimal solutions.
 • Benchmarking reveals significant computational efficiency. For batch sampling, TrajDiffuser drastically reduces runtime compared to numerical optimization—up to 86% reduction for large sample sizes.
 • The framework’s compositional capability is demonstrated via C-TrajDiffuser. This variant composes additional constraints – for example, by incorporating state-triggered energy functions to enforce the velocity-triggered angle-of-attack limit and even to include drag effects from a lower-fidelity 3 DoF model. Both product and negation compositions are tested, with negation composition showing very low constraint violation rates.

7. Conclusions and Future Work:
 • TrajDiffuser is presented as the first diffusion model tailored for powered descent guidance. Its ability to generate concurrent, multi-modal trajectories while handling diverse constraints through compositional methods is emphasized.
 • The approach provides an efficient warm-start for non-convex trajectory optimizations, thereby reducing convergence times while maintaining acceptable optimality.
 • The paper also provides theoretical bounds on the reverse diffusion error when composing energy functions, offering insights into accuracy trade-offs.
 • Future work includes further tightening these error bounds, extending models to incorporate even more constraints (such as additional aerodynamic and kinematic limits), and leveraging MCMC for improved sampling fidelity.

In essence, this work lays the foundation for a compositional trajectory generation toolbox that is generalizable, computationally efficient, and adaptable to complex real-world constraints, addressing a critical need in autonomous spacecraft guidance.

---

# R2-Diff: Denoising by diffusion as a refinement of retrieved motion for image-based motion prediction

**Authors:** Takeru Oba, Norimichi Ukita

**Year:** 2023

This paper proposes R2-Diff, a novel image-based motion prediction method for robot manipulation that integrates retrieval and diffusion. The key insight of the work is to refine a motion retrieved from a dataset—based on the similarity of image features—using a diffusion model rather than generating a motion from randomly sampled Gaussian noise. This approach directly addresses two critical challenges of traditional diffusion models in motion prediction: (1) the randomness of the initial sample and (2) the difficulty in early denoising steps when the initial sample is far from an appropriate motion.

Key details of the work include:

1. Problem Setting and Motivation:
 • In image-based motion prediction, a robot must generate a sequence of motions (positions, rotations, and grasping states) that successfully accomplish a task given an input RGB-D image of the environment.
 • Traditional diffusion models start with a Gaussian noise sample, which may lead to contextually inappropriate motions. In contrast, R2-Diff retrieves an initial motion close to the desired motion and then refines it.

2. R2-Diff Methodology:
 • Retrieval: The method uses an image-based retrieval system that computes the similarity between a test image and training images by extracting features along the motion trajectory. A Spatially-aligned Temporal Encoding (STE) module is employed to extract these motion-dependent image features, ensuring that only the most relevant regions (e.g., around the robot hand) are considered.
 • Diffusion Refinement: Once a motion is retrieved from the dataset, it is noised (using a modified noise injection process) before being refined through a reverse diffusion process. The diffusion model is trained to gradually denoise this input, thereby refining the retrieved motion to better match the test image context.
 • Hyperparameter Tuning: A key contribution is a novel automatic tuning method for the noise scale in the diffusion process. By calculating the distance between a ground-truth motion and its nearest neighbor in the training set (weighted separately for position, rotation, and grasping), the method adjusts the noise injection parameters so that the retrieved motions fall within a “learnable” distribution. This tuning minimizes the excessive noise that traditional diffusion models would introduce, thereby optimizing the model specifically for motion refinement scenarios.

3. Model Architecture:
 • The architecture builds upon components such as a U-Net-like image encoder, the STE module for extracting motion-dependent features, and a Transformer module to integrate motion, image, and timestep features.
 • Rather than relying solely on a fixed Gaussian sampling, the architecture accepts the retrieved motion, adds controlled noise, and then processes it through the network to predict the residuals used for denoising.

4. Experimental Validation:
 • The authors validate their method on 16 different robot manipulation tasks from the RLBench benchmark. These tasks include various challenging manipulations such as picking up objects, pressing buttons, and stacking items.
 • Comparisons with state-of-the-art methods (like RT1, VINN, DMO-EBM, and Diffusion Policy) show that R2-Diff achieves improved success rates—in some cases substantially higher—by effectively combining retrieval with diffusion-based refinement.
 • Detailed ablation studies explore the impact of different retrieval strategies (STE versus other retrieval methods), the tuning of noise parameters (varying the rank of the nearest neighbor used), and the number of diffusion steps. These analyses demonstrate that starting from a contextually relevant retrieved motion and optimally tuning the diffusion process leads to more accurate and robust motion predictions.

5. Limitations and Future Work:
 • The paper notes that in some cases the refinement process can actually decrease success rates, especially when the retrieved motion is already near optimal. This mismatch occurs because the noise distribution used for tuning (based on the maximum error) does not always accurately reflect the true distribution of retrieved motions.
 • Future work is focused on further adjusting the tuning process to better approximate the actual distribution of retrieved motions and thereby improve performance further.

In summary, R2-Diff innovatively refines a retrieved motion through a diffusion process tuned for motion refinement. By leveraging image-based retrieval with targeted feature extraction and carefully adjusting the diffusion noise parameters, the method significantly improves motion prediction in robotic manipulation tasks over existing state-of-the-art approaches.

---

# DiffRoad: Realistic and Diverse Road Scenario Generation for Autonomous Vehicle Testing

**Authors:** Junjie Zhou, Lin Wang, Qiang Meng, Xiaofan Wang

**Year:** 2024

This paper introduces DiffRoad—a novel method that leverages diffusion probabilistic models to generate realistic, diverse, and controllable 3D road scenarios for autonomous vehicle (AV) testing. The central idea is to transform white noise into high-fidelity road layouts by progressively denoising the noise through a reverse diffusion process, while preserving essential spatial properties of real-world roads. Key aspects of the paper include:

1. Purpose and Motivation:
 • Recognizing the limitations of current road scenario generation methods (manual creation, aerial imagery conversion, limited variety) which restrict large‐scale and diverse AV simulation tests, the work aims to build a comprehensive and scalable scenario library.
 • A digital twin-based simulation test environment demands road scenarios in formats such as OpenDRIVE, allowing realistic testing of autonomous driving systems.

2. Core Contributions:
 • DiffRoad is proposed as the first framework to apply diffusion models for generating 3D road scenarios, capturing the statistical distribution of real-world road networks.
 • To enable controllability (e.g., specifying road type, lane structure, road scale), a road attribute embedding module based on wide and deep networks is integrated, allowing users to steer the generation process.
 • The authors design a custom network architecture called Road-UNet—which modifies the traditional U-Net by incorporating FreeU operations, specialized skip and backbone feature scaling, and attention mechanisms—to effectively predict noise levels at each diffusion time step and better model spatial dependencies.
 • A hybrid loss function combines mean square error and a smoothness regularization term. The latter ensures the distances between sequential points in the generated scenarios are consistent with real-world data.
 • A scene-level evaluation module is introduced that filters the generated scenarios using metrics like road continuity (capturing curvature transitions) and road reasonableness (avoiding overlapping roads), ensuring only realistic and compliant scenes are selected.
 • Finally, the generated road scenarios are automatically converted into the OpenDRIVE format for seamless integration with mainstream simulation platforms (e.g., CARLA, VTD).

3. Methodology Overview:
 • The model uses a forward diffusion process to gradually add Gaussian noise to real-world road data and then a reverse diffusion process, guided by conditional road attributes, to remove the noise and recreate the road layout.
 • Road-UNet plays a crucial role in this process by fusing multi-scale features and applying frequency-specific adjustments (via Fourier domain masking) to enhance details and smoothness.
 • Controllable generation is achieved by embedding road attributes such as road type, scale, and structure into the diffusion model, thereby producing scenarios that meet desired specifications.

4. Experimental Validation:
 • The method is evaluated on real-world datasets collected from Singapore, the United Kingdom, and Japan. These data cover diverse road types like intersections, roundabouts, pick-up and drop-off (PUDO) zones, and flyovers.
 • Various metrics—including Hausdorff Distance for realism, Jensen-Shannon Divergence for distribution similarity (both in road lengths and relative control point distances), and a smoothness measure based on the second-order derivative—are used to compare the generated scenarios with real-world data.
 • Ablation studies demonstrate the contributions of the Road-UNet module, the conditional attribute embedding, the smoothness loss, and the scene evaluation module. The complete DiffRoad model consistently outperforms baseline generative models (VAE and GAN) as well as various ablated versions.
 • Additionally, efficiency improvements are showcased with DiffRoad generating scenarios orders of magnitude faster than existing methods (e.g., optimized FLYOVER approaches), and the generated scenarios are successfully deployed in AV simulations on CARLA and VTD platforms.

5. Key Insights:
 • DiffRoad bridges the gap between theoretical diffusion models and practical AV simulation needs by transforming noisy data into highly realistic and structured road scenes.
 • The proposed architecture and loss formulation facilitate the generation of scenarios that are both statistically consistent with real data and tailored for specific testing requirements.
 • The combination of conditional generation, enhanced spatial modeling through Road-UNet, and a rigorous scene evaluation process enables the creation of a vast and diverse library that can help accelerate AV testing and inform future road infrastructure designs.

In summary, the paper presents a comprehensive and novel approach to road scenario generation that promises to significantly enhance the reliability and diversity of simulation environments for autonomous vehicle testing. Its integration of diffusion models with specialized network architectures and evaluation modules provides an effective, scalable solution to one of the critical bottlenecks in AV simulation.

---

# Diffusion Meets DAgger: Supercharging Eye-in-hand Imitation Learning

**Authors:** Xiaoyu Zhang, Matthew Chang, Pranav Kumar, Saurabh Gupta

**Year:** 2024

“Diffusion Meets DAgger: Supercharging Eye-in-hand Imitation Learning” introduces a novel framework—DMD—that significantly improves the robustness and sample efficiency of imitation learning for robotic manipulation tasks using an eye-in-hand camera setup. The central challenge the paper addresses is the compounding execution error problem, where small mistakes during policy execution lead the robot into states not covered by the expert demonstrations. Instead of relying on expensive human-supervised data collection as in traditional DAgger, DMD automatically synthesizes out-of-distribution observations along with corrective action labels, thereby augmenting the training data.

Key to the technique is a conditional diffusion model that is fine-tuned from a pretrained model. The model is conditioned on a reference image from an expert demonstration and a small transformation (∆p) representing a perturbation in camera or robot pose. By generating perturbed views—even when the scene undergoes deformations due to manipulation—the system creates realistic off-trajectory images. Each of these synthesized images is paired with an action label computed based on relative camera poses obtained via structure from motion. Care is taken to mitigate issues such as the “overshooting problem” by choosing the appropriate future frame (e.g., It+3) to compute labels that still drive the system toward task success.

The paper provides detailed experimental results across four different manipulation tasks—non-prehensile pushing, stacking cups, pouring coffee beans, and hanging a shirt. In each task, DMD substantially outperforms standard behavior cloning. For example, in the pushing task, DMD achieves around an 80% success rate with only eight demonstrations, whereas behavior cloning reaches merely 20%; similar improvements are observed for stacking and pouring tasks. Moreover, DMD outperforms an existing NeRF-based augmentation approach (SPARTN) by generating higher-quality, more consistent synthetic views even when the scene is non-static.

Additional design insights include:
• The use of both task-specific and task-agnostic “play” data to further finetune the diffusion model, demonstrating that even simpler data can boost downstream policy performance.
• An analysis of different augmentation strategies and labeling schemes, showing that careful choice of how synthetic views are labeled (using a further future frame to avoid conflicting supervision) is critical to policy performance.
• A detailed comparison with alternative augmentation methods such as color jitter and flipping, where DMD still shows significant performance gains.

Overall, the paper’s main contribution is the innovative integration of state-of-the-art diffusion models with imitation learning. By synthesizing out-of-distribution images and associating them with corrective action labels, DMD reduces the reliance on extensive expert supervision and enhances the robot’s ability to recover from errors during real-world execution. This work paves the way for more robust and sample-efficient learning in robotic manipulation, with potential extensions to more complex object dynamics and additional sensory modalities.

---

# DexDiffuser: Generating Dexterous Grasps with Diffusion Models

**Authors:** Zehang Weng, Haofei Lu, Danica Kragic, Jens Lundell

**Year:** 2024

The paper “DexDiffuser: Generating Dexterous Grasps with Diffusion Models” introduces a novel, data-driven approach for dexterous grasping that works directly on partial object point clouds—overcoming a key limitation of many previous methods that require complete object views. The authors propose a two-part framework composed of the conditional diffusion-based grasp sampler, called DexSampler, and the dexterous grasp evaluator, named DexEvaluator.

DexSampler is built on a diffusion model that progressively “denoises” randomly sampled grasp candidates. Conditioned on object features extracted from a partial point cloud (encoded using the Basis Point Set representation), the model learns to generate high-quality grasps in a high-dimensional 16-DoF space that includes both the SE(3) pose and the multi-finger joint configurations. The diffusion process involves a forward process that gradually adds Gaussian noise and a learned reverse process that reconstructs the successful grasp by removing the noise in an iterative manner.

DexEvaluator, on the other hand, scores the generated grasps by predicting their success probability. To improve sensitivity to small yet crucial changes in the grasp configuration, the authors incorporate a frequency encoding mechanism before processing the grasp parameters. This evaluator not only ranks grasps for execution but also provides guidance for two grasp refinement techniques. The first, Evaluator-Guided Diffusion (EGD), integrates the gradient from the evaluator into the inverse diffusion process to steer generated grasps toward more promising regions. The second, Evaluator-based Sampling Refinement (ESR), improves already sampled grasps by locally optimizing the parameters via a Metropolis-Hasting algorithm. A two-stage ESR (ESR-2) is also proposed to refine the global grasp parameters (position and rotation) first before fine-tuning the finger joint configurations.

The models are trained on a comprehensive dataset generated from DexGraspNet, which includes 1.7 million grasps (both successful and unsuccessful) across more than 5000 objects. This large dataset, along with simulated point clouds from IsaacGym, allows the method to learn to generate and evaluate grasps in highly variable scenarios.

Experimental evaluations are presented in simulation and on a real robot. In simulation, DexDiffuser consistently outperforms state-of-the-art methods such as FFHNet and UniDexGrasp. Using both point cloud encodings (BPS and a PointNet++ alternative), the results show that the BPS encoding is more robust to irregularities in the data. Quantitative comparisons demonstrate that DexDiffuser achieves higher grasp success rates (with improvements of about 9% in simulation) and yields more diverse, physically feasible grasps with lower penetration depth. Furthermore, grasp refinement strategies (EGD and ESR) offer marginal yet consistent improvements, especially when applied in sequence.

The real-world experiments involve a Franka Panda robot equipped with an Allegro Hand and a Kinect V3 sensor. Despite the challenges of noisy point cloud data and unmodeled environmental constraints, DexDiffuser achieves an average success rate nearly 20% higher than FFHNet. Trials on a set of nine diverse objects further validated the method’s effectiveness, even though the diffusion-based criterion leads to slower inference times due to the iterative denoising process.

In summary, the paper presents a robust framework for dexterous grasp generation on partial observations using diffusion models. By combining a diffusion-based grasp sampler with an evaluator and dedicated refinement strategies, DexDiffuser significantly enhances grasp success rates in both simulated and real-world environments. The work not only demonstrates improvements in grasp quality and diversity but also opens up promising avenues for addressing other complex dexterous manipulation challenges despite limitations such as the sim-to-real gap and increased sampling time.

---

# Layout Sequence Prediction From Noisy Mobile Modality

**Authors:** Haichao Zhang, Yi Xu, Hongsheng Lu, Takayuki Shimizu, Yun Fu

**Year:** 2023

The paper “Layout Sequence Prediction From Noisy Mobile Modality” introduces a novel framework—LTrajDiff—to predict pedestrian bounding box trajectories (layout sequences) by fusing incomplete, obstructed visual data with noisy mobile sensor signals (e.g., from IMU, Wi-Fi, and other mobile sensors). The major motivation is that existing trajectory predictors rely on long, complete, and unobstructed visual sequences captured by cameras. However, in real-world scenarios, cameras often encounter blind spots, occlusions, or missing observations. By incorporating mobile sensor data, which does not suffer from “out-of-sight” issues, the authors aim to overcome the limitations of pure vision-based systems.

Key Contributions and Methodology:
1. LTrajDiff Framework:  
 • The proposed model uses a denoising diffusion process to transform noisy mobile data into accurate 3D layout sequences (bounding boxes with depth, size, and spatial projection).  
 • The method implicitly infers missing layout details like object size and projection orientation from extremely short or heavily obstructed visual input.

2. Module Design:  
 • Random Mask Strategy (RMS): This module simulates realistic obstructions by randomly masking input visual sequences (varying both location and ratio), so the network learns to handle incomplete data during training.  
 • Siamese Masked Encoding Module: Divided into two submodules, it processes the dual modalities. The Temporal Alignment Module (TAM) synchronizes and aligns noisy mobile signals with sparse visual observations using a transformer-based approach. The Layout Extraction Module (LEM) focuses on recovering spatial layout and object size details from the unmasked portions of the visual input.  
 • Modality Fusion Module (MFM): This module fuses the aligned temporal features from TAM with the spatial features from LEM through fully connected layers, mapping the joint representation into parameters that guide the denoising diffusion decoder.

3. Denoising Diffusion Decoder:  
 • Implemented using a transformer decoder, this component iteratively refines the noisy input through a coarse-to-fine diffusion process. At each iteration, the model predicts and removes added noise until a high-quality layout sequence is synthesized.  
 • Training is done in an end-to-end manner with an L2 loss comparing predicted noise with the added noise, providing robustness even when only a single or very few timestamps are visible in the input.

Experiments and Insights:
• The paper evaluates the method on two datasets (Vi-Fi and H3D) that include both visual layout sequences and wireless sensor signals.
• Two novel evaluation metrics are proposed:
 – Mean Square Error per Timestamp (MSE-T): Measures the average error in layout predictions.
 – Intersection over Union with Depth (IoU-D): Extends standard IoU by incorporating depth information, reflecting the quality of 3D bounding box predictions.
• Quantitative experiments demonstrate that LTrajDiff outperforms adapted baselines such as ViTag, UNet, Transformer, and LSTM in both randomly obstructed and extremely short input sequence scenarios.
• Ablation studies confirm that each module (RMS, TAM, LEM, and MFM) plays a critical role. Removing any module causes a significant drop in performance, validating the importance of effective modality fusion and robust feature extraction.
• Additional experiments isolating the visual and mobile modalities show that both are vital and contribute comparably to successful layout sequence prediction.

Conclusion:
The work pioneers the fusion of noisy mobile data with incomplete visual input to achieve accurate layout sequence (trajectory) prediction, addressing a critical challenge in applications such as autonomous driving and robotics. By leveraging a denoising diffusion framework combined with carefully designed encoding and fusion modules, the proposed method demonstrates state-of-the-art results in predicting pedestrian bounding boxes even under severe occlusions and extremely limited visual data. This approach opens new avenues for robust trajectory prediction by exploiting complementary sensor modalities in real-world settings.

---

