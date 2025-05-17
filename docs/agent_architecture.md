# Lightweight Agentic Runtime Architecture

## Core Components

1. **Registry**: Central repository of agents, tools, and policies
2. **Agent Manager**: Handles agent lifecycle (creation, execution, termination)
3. **Policy Engine**: Enforces access control between agents and tools
4. **Prompt Store**: Manages versioned prompts in YAML format
5. **Discovery Service**: Enables agents to find other agents by capability
6. **Execution Engine**: Runs agent workflows and tool executions
7. **Observer**: Monitors and logs agent activities (optional)

## Component Interactions

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│             │      │             │      │             │
│  Registry   │◄────►│Agent Manager│◄────►│Policy Engine│
│             │      │             │      │             │
└─────┬───────┘      └──────┬──────┘      └─────────────┘
      │                     │
      │                     │
┌─────▼───────┐      ┌──────▼──────┐
│             │      │             │
│ Prompt Store│◄────►│   Agents    │
│             │      │             │
└─────────────┘      └──────┬──────┘
                            │
                            │
                     ┌──────▼──────┐      ┌─────────────┐
                     │             │      │             │
                     │    Tools    │◄────►│  Observer   │
                     │             │      │             │
                     └─────────────┘      └─────────────┘
```

