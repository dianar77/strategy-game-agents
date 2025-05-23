# agent_code.py
"""
This script provides the get_prompt_enhancement function, which offers strategic advice 
to the LLM playing Catan. The strategy implementation is based on effective gameplay tactics.
"""

def get_prompt_enhancement():
    """
    Returns a string to be added to the beginning of the LLM prompt.
    It offers strategic guidance to increase the player's chances of winning.
    """

    instructions = (
        "Catan Strategy Guide:\n"
        "1. Prioritize high-probability tiles with numbers like 6 and 8 for \\"
        "initial settlements. Balance the spread of resources.\n"
        "2. Cities are a priority: Upgrade to cities at the earliest chance \\"
        "to maximize the resource collection rate.\n"
        "3. Diversify resources to ensure consistent access to all. \\"
        "Supplement this with ports if needed.\n"
        "4. Target the leading opponent with robbers to control their growth. \\"
        "Collaborate trades to discourage runaway players.\n"
        "5. Aim for Longest Road or Largest Army for the additional 2 Victory Points.\n"
        "6. Develop a balanced use of development cards: use Knights for Largest Army \\"
        "and consider Year of Plenty/Monopoly cards strategically.\n"
        "7. Avoid overcommitting to roads to prevent resource leakage unless it \\"
        "contributes directly to settlements or Longest Road.\n"
        "8. Trade wisely: Influence trades to your advantage while being mindful \\"
        "of the competition's status.\n"
        "9. Hoard ore and wheat during early and mid-game to maintain a \\"
        "lead in building cities or buying critical development cards.\n"
        "10. Adapt priorities based on opponents' moves. For example, \\"
        "block critical resources of a runaway player or out-position their \\"
        "road network structurally.\n"
        "\n"
    )

    return instructions