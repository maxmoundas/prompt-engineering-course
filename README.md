# Prompt Engineering: Leveraging Large Language Models (LLMs)

A comprehensive four-day course on large language models, prompt engineering, and advanced AI applications designed for STEM researchers and practitioners.

## Course Overview

This course provides a thorough introduction to Large Language Models and practical prompt engineering techniques for research applications. The curriculum progresses from foundational concepts to advanced implementation strategies, emphasizing real-world applications, ethical considerations, and emerging technologies in the rapidly evolving AI landscape.

## Interactive Demos & Resources

**Hands-on examples from class sessions (all with Streamlit interfaces):**

- **[LLM Parameter Explorer](day2/llm-parameter-explorer/)** - Experiment with LLM parameters (temperature, top_p, model selection)
- **[Agent Framework](https://github.com/maxmoundas/agent-framework)** - Autonomous agent with tool calling system (generate QR code, fetch recent news, send email via Gmail connector) - we built the QR code tool as a live demo demonstrating Cursor's LLM-assisted software engineering capabilities in class
- **[RAG System](https://github.com/maxmoundas/RAG)** - Retrieval-Augmented Generation with document embedding

*These demos provide practical, runnable examples of the concepts covered in class. Feel free to clone, modify, and experiment with them.*

### Target Audience

PhD researchers and post-docs from STEM disciplines seeking to understand and leverage generative AI technologies in their research workflows.

### Prerequisites

- Basic familiarity with computational concepts
- No prior experience with AI or machine learning required
- Interest in applying AI tools to research contexts

## Course Structure

The course is organized into four sessions covering foundational through advanced topics:

### Day 1: Foundations of LLMs (Monday, July 14)
**Core Topics:**
- What are Large Language Models? Architecture and capabilities
- How LLMs work: Transformers, attention mechanisms, and training processes
- Key terminology: Tokens, context windows, temperature, hallucinations
- Major LLM providers and ecosystem (OpenAI, Anthropic, Google, Meta, AWS)
- Current best models and evaluation methodologies
- Ethical considerations: Copyright, environmental impact, bias, and misuse
- Understanding limitations: Reasoning, accuracy, and the "black box" problem

**Key Learning Outcomes:**
- Understand LLM capabilities and fundamental limitations
- Recognize the difference between language understanding and reasoning
- Appreciate ethical implications and responsible use practices
- Identify appropriate use cases for different models and providers

### Day 2: Prompt Engineering Fundamentals (Tuesday, July 15)
**Core Topics:**
- What is a prompt? Understanding the interface between human intent and model behavior
- Anatomy of effective prompts: Clarity, context, formatting guidance, demonstrations, constraints, and evaluation criteria
- All influences on LLM output: User prompts, system prompts, fine-tuning, temperature settings, and external tools
- Essential prompt patterns and techniques:
  - Meta-prompting: Using LLMs to improve your prompts
  - Flipped interaction pattern: Enabling LLMs to ask clarifying questions
  - Few-shot prompting: Learning from examples
  - Chain-of-thought (CoT): Step-by-step reasoning
  - Role-based prompting: Persona patterns and expert roles
  - ReAct prompting: Combining reasoning and action
  - Conformal abstention: When to say "I don't know"
  - Structured data extraction: Converting unstructured content to structured formats
- Common prompting mistakes and how to avoid them
- Large Language Models vs. Large Reasoning Models (LRMs): Understanding when to use each
- Security and safety considerations: Prompt injection attacks, jailbreaking, guardrails, and system prompt protection

**Key Learning Outcomes:**
- Master the fundamentals of effective prompt design and structure
- Apply proven prompt patterns to solve real-world problems
- Understand the security vulnerabilities inherent in LLM systems
- Distinguish between appropriate use cases for LLMs versus LRMs
- Develop systematic approaches to prompt engineering rather than trial-and-error
- Recognize and mitigate common prompt engineering pitfalls

### Day 3: RAG & Multimodal LLMs (Thursday, July 17)
**Core Topics:**

**Retrieval-Augmented Generation (RAG):**
- Understanding RAG: Augmenting LLMs with external information retrieval
- RAG architecture: Embedders, vector databases, and retrieval systems
- Basic RAG implementation: Document embedding, query processing, and context injection
- Advanced techniques: RAG Fusion for comprehensive multi-perspective retrieval
- Citations and source grounding: Ensuring transparency and reducing hallucinations
- Large document processing: Chunked retrieval and semantic search strategies

**AI Assistants and Research Tools:**
- Building domain-specific assistants: Custom system instructions and knowledge bases
- Deep Research capabilities: Multi-faceted inquiry with extensive source citation
- Abstracting prompting complexity from end users

**Multimodal Capabilities:**
- Visual understanding: Image interpretation, OCR, and structured data extraction
- Voice and video interfaces: Real-time interaction and accessibility applications
- Content generation across modalities:
  - Image generation: Photorealistic and artistic content creation
  - Voice generation: Text-to-speech synthesis and voice cloning
  - Video generation: State-of-the-art AI video creation with synchronized audio

**Practical Applications:**
- Document analysis and summarization workflows
- Multimodal research applications and accessibility tools
- Integration strategies for research workflows

**Key Learning Outcomes:**
- Implement RAG systems for grounded, source-backed AI responses
- Design and deploy custom AI assistants for domain-specific research tasks
- Leverage multimodal AI capabilities for diverse content creation and analysis
- Understand the current state-of-the-art in AI-generated media across text, image, voice, and video
- Apply RAG and multimodal techniques to enhance research productivity and accuracy
- Recognize opportunities for multimodal AI integration in STEM research contexts

### Day 4: Agents & LLM-Assisted Software Engineering (Friday, July 18)
**Core Topics:**

**Introduction to autonomous agents: From reactive LLMs to proactive systems**
- Agent architecture: LLM core, toolsets, memory systems, and planning components
- The agent loop: Observe → Think → Act → Reflect cycles
- Tool and function calling: Structured outputs and external API integration
- Agent memory systems: Short-term task history and long-term vector databases
- Multi-agent systems: Collaboration, specialization, and coordination patterns
- Agentic misalignment: Understanding risks and safety considerations in autonomous systems
- Model Control Protocol (MCP): Standardizing agent-tool interactions

**LLM-powered software engineering tools: Cursor, Claude Code, and contextual code understanding**
- Challenges in LLM-based development: Code quality, security, and scalability considerations
- Vibe coding: Flow-based development approaches and their trade-offs
- Current state and future of AGI: Definitions, timelines, and industry perspectives

**Key Learning Outcomes:**
- Design and implement basic autonomous agent architectures
- Understand the perception → plan → act → reflect cycle for agent behavior
- Recognize the capabilities and limitations of current AI coding tools
- Evaluate when to use agents versus traditional programming approaches
- Understand the safety and alignment challenges in autonomous AI systems
- Distinguish between productivity amplification and wholesale replacement in software engineering
- Apply agent-based approaches to appropriate research and development workflows

## Instructor

**Max Moundas**  
Generative AI Research Engineer, Vanderbilt University's Generative AI Center

**Background:**
- B.S. in Computer Science, Vanderbilt University (May 2023)
- One of four core developers of [Amplify](https://www.amplifygenai.org/) - open-source enterprise AI platform
- Researcher in LLM applications and prompt engineering methodologies

## Resources and Links

- [Amplify Platform](https://www.amplifygenai.org/)
- [Chatbot Arena Leaderboard](https://lmarena.ai/leaderboard)
- [Hugging Face Model Hub](https://huggingface.co/)

## Contact

For questions or further information:
- **Email:** maxmoundas@gmail.com
- **LinkedIn:** [https://www.linkedin.com/in/maxmoundas/](https://www.linkedin.com/in/maxmoundas/)
- **Website:** [maxmoundas.com](https://maxmoundas.com)

---

*Course materials updated July 18, 2025*