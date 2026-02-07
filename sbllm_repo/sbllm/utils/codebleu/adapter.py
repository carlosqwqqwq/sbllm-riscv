import logging
import tree_sitter
from tree_sitter import Language, Parser

logger = logging.getLogger(__name__)

def get_language(ptr, name):
    """
    Robustly create a Language object across tree-sitter versions.
    v0.25+: Language(ptr, name)
    v0.22+: Language(ptr)
    old: Language(path, name)
    """
    try:
        # Try new API first (v0.25+) which requires name
        return Language(ptr, name)
    except TypeError:
        pass
    except Exception as e:
        logger.debug(f"Language(ptr, name) failed: {e}")

    try:
        # Fallback to v0.22+ (capsule only)
        return Language(ptr)
    except TypeError:
        pass
    except Exception as e:
        logger.debug(f"Language(ptr) failed: {e}")
        
    raise ValueError("Could not instantiate tree_sitter.Language with provided pointer")

def get_parser(lang):
    """
    Robustly create a Parser object across tree-sitter versions.
    v0.25+: Parser(lang) - set_language removed
    old: Parser(); parser.set_language(lang)
    """
    try:
        # Try new API first (v0.25+)
        return Parser(lang)
    except TypeError:
        # If Parser() works but Parser(lang) fails, it might be old API
        pass
    except Exception as e:
        logger.debug(f"Parser(lang) failed: {e}")

    # Fallback to old API
    try:
        parser = Parser()
        if hasattr(parser, 'set_language'):
            parser.set_language(lang)
        else:
            # Try property setter
            parser.language = lang
        return parser
    except Exception as e:
        logger.error(f"Failed to create Parser with fallback: {e}")
        raise
