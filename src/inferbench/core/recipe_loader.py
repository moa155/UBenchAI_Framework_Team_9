"""
Recipe Loader for InferBench Framework.

Handles loading, parsing, and validating YAML recipe files for
servers, clients, monitors, and benchmarks.
"""

from pathlib import Path
from typing import Optional, Type, TypeVar
import yaml

from pydantic import ValidationError

from inferbench.core.config import get_config
from inferbench.core.exceptions import (
    RecipeNotFoundError,
    RecipeParseError,
    RecipeValidationError,
)
from inferbench.core.models import (
    BaseRecipe,
    ServerRecipe,
    ClientRecipe,
    MonitorRecipe,
    RecipeType,
)
from inferbench.utils.logging import get_logger

logger = get_logger(__name__)

# Type variable for recipe types
T = TypeVar("T", bound=BaseRecipe)


class RecipeLoader:
    """
    Loads and validates recipe files from the recipes directory.
    
    Supports lazy loading and caching of recipes for performance.
    """
    
    # Map recipe types to their model classes
    RECIPE_MODELS: dict[RecipeType, Type[BaseRecipe]] = {
        RecipeType.SERVER: ServerRecipe,
        RecipeType.CLIENT: ClientRecipe,
        RecipeType.MONITOR: MonitorRecipe,
    }
    
    # Map recipe types to their directory names
    RECIPE_DIRS: dict[RecipeType, str] = {
        RecipeType.SERVER: "servers",
        RecipeType.CLIENT: "clients",
        RecipeType.MONITOR: "monitors",
        RecipeType.BENCHMARK: "benchmarks",
    }
    
    def __init__(self, recipes_dir: Optional[Path] = None):
        """
        Initialize the recipe loader.
        
        Args:
            recipes_dir: Path to recipes directory. Uses config default if not specified.
        """
        self.recipes_dir = recipes_dir or get_config().recipes_dir
        self._cache: dict[str, BaseRecipe] = {}
        logger.debug(f"RecipeLoader initialized with recipes_dir: {self.recipes_dir}")
    
    def _get_recipe_path(self, recipe_type: RecipeType, recipe_name: str) -> Path:
        """Get the full path to a recipe file."""
        type_dir = self.RECIPE_DIRS.get(recipe_type, recipe_type.value)
        
        # Try with .yaml extension
        recipe_path = self.recipes_dir / type_dir / f"{recipe_name}.yaml"
        if recipe_path.exists():
            return recipe_path
        
        # Try with .yml extension
        recipe_path = self.recipes_dir / type_dir / f"{recipe_name}.yml"
        if recipe_path.exists():
            return recipe_path
        
        raise RecipeNotFoundError(recipe_name, recipe_type.value)
    
    def _parse_yaml(self, recipe_path: Path) -> dict:
        """Parse a YAML file and return the data."""
        try:
            with open(recipe_path, "r") as f:
                data = yaml.safe_load(f)
            
            if data is None:
                raise RecipeParseError(str(recipe_path), "Empty YAML file")
            
            return data
        except yaml.YAMLError as e:
            raise RecipeParseError(str(recipe_path), str(e))
        except IOError as e:
            raise RecipeParseError(str(recipe_path), f"IO error: {e}")
    
    def _validate_recipe(
        self, 
        data: dict, 
        recipe_type: RecipeType,
        recipe_name: str
    ) -> BaseRecipe:
        """Validate recipe data against the appropriate Pydantic model."""
        model_class = self.RECIPE_MODELS.get(recipe_type)
        
        if model_class is None:
            # For benchmark recipes, return as base recipe for now
            return BaseRecipe(**data)
        
        try:
            return model_class(**data)
        except ValidationError as e:
            errors = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
            raise RecipeValidationError(recipe_name, errors)
    
    def load(
        self, 
        recipe_type: RecipeType, 
        recipe_name: str,
        use_cache: bool = True
    ) -> BaseRecipe:
        """
        Load a recipe by type and name.
        
        Args:
            recipe_type: Type of recipe to load
            recipe_name: Name of the recipe
            use_cache: Whether to use cached recipe if available
            
        Returns:
            Validated recipe object
            
        Raises:
            RecipeNotFoundError: If recipe file doesn't exist
            RecipeParseError: If YAML parsing fails
            RecipeValidationError: If validation fails
        """
        cache_key = f"{recipe_type.value}:{recipe_name}"
        
        # Check cache
        if use_cache and cache_key in self._cache:
            logger.debug(f"Using cached recipe: {cache_key}")
            return self._cache[cache_key]
        
        # Find and load recipe
        recipe_path = self._get_recipe_path(recipe_type, recipe_name)
        logger.debug(f"Loading recipe from: {recipe_path}")
        
        # Parse YAML
        data = self._parse_yaml(recipe_path)
        
        # Ensure type is set correctly
        data["type"] = recipe_type.value
        if "name" not in data:
            data["name"] = recipe_name
        
        # Validate and create recipe object
        recipe = self._validate_recipe(data, recipe_type, recipe_name)
        
        # Cache the recipe
        self._cache[cache_key] = recipe
        logger.info(f"Loaded recipe: {recipe_name} ({recipe_type.value})")
        
        return recipe
    
    def load_server(self, recipe_name: str, use_cache: bool = True) -> ServerRecipe:
        """Load a server recipe."""
        recipe = self.load(RecipeType.SERVER, recipe_name, use_cache)
        assert isinstance(recipe, ServerRecipe)
        return recipe
    
    def load_client(self, recipe_name: str, use_cache: bool = True) -> ClientRecipe:
        """Load a client recipe."""
        recipe = self.load(RecipeType.CLIENT, recipe_name, use_cache)
        assert isinstance(recipe, ClientRecipe)
        return recipe
    
    def load_monitor(self, recipe_name: str, use_cache: bool = True) -> MonitorRecipe:
        """Load a monitor recipe."""
        recipe = self.load(RecipeType.MONITOR, recipe_name, use_cache)
        assert isinstance(recipe, MonitorRecipe)
        return recipe
    
    def list_recipes(self, recipe_type: RecipeType) -> list[str]:
        """
        List all available recipes of a given type.
        
        Args:
            recipe_type: Type of recipes to list
            
        Returns:
            List of recipe names
        """
        type_dir = self.RECIPE_DIRS.get(recipe_type, recipe_type.value)
        recipes_path = self.recipes_dir / type_dir
        
        if not recipes_path.exists():
            logger.warning(f"Recipes directory not found: {recipes_path}")
            return []
        
        recipes = []
        for file_path in recipes_path.iterdir():
            if file_path.suffix in [".yaml", ".yml"]:
                recipes.append(file_path.stem)
        
        return sorted(recipes)
    
    def list_all(self) -> dict[str, list[str]]:
        """
        List all available recipes organized by type.
        
        Returns:
            Dictionary mapping recipe types to lists of recipe names
        """
        return {
            recipe_type.value: self.list_recipes(recipe_type)
            for recipe_type in RecipeType
        }
    
    def validate_recipe_file(self, recipe_path: Path) -> tuple[bool, list[str]]:
        """
        Validate a recipe file without loading it into cache.
        
        Args:
            recipe_path: Path to recipe file
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        try:
            data = self._parse_yaml(recipe_path)
            
            # Determine recipe type
            recipe_type_str = data.get("type", "server")
            try:
                recipe_type = RecipeType(recipe_type_str)
            except ValueError:
                errors.append(f"Invalid recipe type: {recipe_type_str}")
                return False, errors
            
            # Validate
            self._validate_recipe(data, recipe_type, recipe_path.stem)
            return True, []
            
        except RecipeParseError as e:
            errors.append(str(e))
        except RecipeValidationError as e:
            errors.extend(e.details.get("errors", [str(e)]))
        except Exception as e:
            errors.append(f"Unexpected error: {e}")
        
        return False, errors
    
    def clear_cache(self) -> None:
        """Clear the recipe cache."""
        self._cache.clear()
        logger.debug("Recipe cache cleared")
    
    def reload_recipes(self) -> None:
        """Clear cache and reload all recipes."""
        self.clear_cache()
        
        for recipe_type in RecipeType:
            for recipe_name in self.list_recipes(recipe_type):
                try:
                    self.load(recipe_type, recipe_name, use_cache=True)
                except Exception as e:
                    logger.warning(f"Failed to reload recipe {recipe_name}: {e}")


# Global recipe loader instance
_loader: Optional[RecipeLoader] = None


def get_recipe_loader() -> RecipeLoader:
    """Get the global recipe loader instance."""
    global _loader
    if _loader is None:
        _loader = RecipeLoader()
    return _loader
