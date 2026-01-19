"""
Tests for the recipe loader module.
"""

import pytest
from pathlib import Path

from inferbench.core.recipe_loader import RecipeLoader, get_recipe_loader
from inferbench.core.models import RecipeType, ServerRecipe, ClientRecipe
from inferbench.core.exceptions import RecipeNotFoundError, RecipeValidationError, RecipeParseError


class TestRecipeLoader:
    """Tests for RecipeLoader class."""
    
    @pytest.fixture
    def loader(self, tmp_path):
        """Create a recipe loader with temporary directory."""
        recipes_dir = tmp_path / "recipes"
        recipes_dir.mkdir()
        (recipes_dir / "servers").mkdir()
        (recipes_dir / "clients").mkdir()
        (recipes_dir / "monitors").mkdir()
        return RecipeLoader(recipes_dir)
    
    @pytest.fixture
    def sample_server_recipe(self, tmp_path):
        """Create a sample server recipe file."""
        recipes_dir = tmp_path / "recipes" / "servers"
        recipes_dir.mkdir(parents=True, exist_ok=True)
        
        recipe_content = """
name: test-server
type: server
description: Test server recipe
container:
  image: /path/to/image.sif
  runtime: apptainer
  binds:
    - /data:/data
resources:
  nodes: 1
  gpus: 1
  memory: 32G
  time: "02:00:00"
network:
  ports:
    - name: api
      port: 8000
      protocol: http
environment:
  MODEL_NAME: test-model
command: python -m server
healthcheck:
  enabled: true
  endpoint: /health
  port: 8000
"""
        recipe_file = recipes_dir / "test-server.yaml"
        recipe_file.write_text(recipe_content)
        return recipe_file
    
    @pytest.fixture
    def sample_client_recipe(self, tmp_path):
        """Create a sample client recipe file."""
        recipes_dir = tmp_path / "recipes" / "clients"
        recipes_dir.mkdir(parents=True, exist_ok=True)
        
        recipe_content = """
name: test-client
type: client
description: Test client recipe
resources:
  nodes: 1
  gpus: 0
  memory: 16G
target:
  url: http://localhost:8000
workload:
  type: stress-test
  requests: 1000
"""
        recipe_file = recipes_dir / "test-client.yaml"
        recipe_file.write_text(recipe_content)
        return recipe_file
    
    def test_load_server_recipe(self, loader, sample_server_recipe):
        """Should load and validate a server recipe."""
        recipe = loader.load_server("test-server")
        
        assert isinstance(recipe, ServerRecipe)
        assert recipe.name == "test-server"
        assert recipe.container.image == "/path/to/image.sif"
        assert recipe.resources.gpus == 1
        assert recipe.resources.memory == "32G"
        assert recipe.environment["MODEL_NAME"] == "test-model"
    
    def test_load_client_recipe(self, loader, sample_client_recipe):
        """Should load and validate a client recipe."""
        recipe = loader.load_client("test-client")
        
        assert isinstance(recipe, ClientRecipe)
        assert recipe.name == "test-client"
        assert recipe.resources.memory == "16G"
    
    def test_recipe_not_found(self, loader):
        """Should raise RecipeNotFoundError for missing recipe."""
        with pytest.raises(RecipeNotFoundError) as exc_info:
            loader.load_server("nonexistent")
        
        assert "nonexistent" in str(exc_info.value)
    
    def test_invalid_yaml(self, loader, tmp_path):
        """Should raise RecipeParseError for invalid YAML."""
        recipes_dir = tmp_path / "recipes" / "servers"
        recipes_dir.mkdir(parents=True, exist_ok=True)
        
        invalid_file = recipes_dir / "invalid.yaml"
        invalid_file.write_text("invalid: yaml: content: [")
        
        with pytest.raises(RecipeParseError):
            loader.load_server("invalid")
    
    def test_validation_error(self, loader, tmp_path):
        """Should raise RecipeValidationError for invalid recipe."""
        recipes_dir = tmp_path / "recipes" / "servers"
        recipes_dir.mkdir(parents=True, exist_ok=True)
        
        # Missing required container field
        invalid_recipe = recipes_dir / "invalid-recipe.yaml"
        invalid_recipe.write_text("""
name: invalid-recipe
type: server
resources:
  memory: invalid-format
""")
        
        with pytest.raises((RecipeValidationError, RecipeParseError)):
            loader.load_server("invalid-recipe")
    
    def test_list_recipes(self, loader, sample_server_recipe, sample_client_recipe):
        """Should list available recipes."""
        server_recipes = loader.list_recipes(RecipeType.SERVER)
        client_recipes = loader.list_recipes(RecipeType.CLIENT)
        
        assert "test-server" in server_recipes
        assert "test-client" in client_recipes
    
    def test_list_all(self, loader, sample_server_recipe, sample_client_recipe):
        """Should list all recipes organized by type."""
        all_recipes = loader.list_all()
        
        assert "server" in all_recipes
        assert "client" in all_recipes
        assert "test-server" in all_recipes["server"]
    
    def test_recipe_caching(self, loader, sample_server_recipe):
        """Should cache loaded recipes."""
        recipe1 = loader.load_server("test-server")
        recipe2 = loader.load_server("test-server")
        
        # Should be same object due to caching
        assert recipe1 is recipe2
    
    def test_cache_bypass(self, loader, sample_server_recipe):
        """Should bypass cache when requested."""
        recipe1 = loader.load_server("test-server", use_cache=True)
        recipe2 = loader.load_server("test-server", use_cache=False)
        
        # Should be different objects
        assert recipe1 is not recipe2
        # But same content
        assert recipe1.name == recipe2.name
    
    def test_clear_cache(self, loader, sample_server_recipe):
        """Should clear recipe cache."""
        loader.load_server("test-server")
        assert len(loader._cache) > 0
        
        loader.clear_cache()
        assert len(loader._cache) == 0
    
    def test_validate_recipe_file(self, loader, sample_server_recipe):
        """Should validate recipe file without caching."""
        is_valid, errors = loader.validate_recipe_file(sample_server_recipe)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_yml_extension(self, loader, tmp_path):
        """Should handle .yml extension."""
        recipes_dir = tmp_path / "recipes" / "servers"
        recipes_dir.mkdir(parents=True, exist_ok=True)
        
        recipe_file = recipes_dir / "yml-recipe.yml"
        recipe_file.write_text("""
name: yml-recipe
type: server
container:
  image: /path/to/image.sif
""")
        
        recipe = loader.load_server("yml-recipe")
        assert recipe.name == "yml-recipe"


class TestResourceSpecValidation:
    """Tests for ResourceSpec validation."""
    
    def test_valid_memory_formats(self):
        """Should accept valid memory formats."""
        from inferbench.core.models import ResourceSpec
        
        for memory in ["16G", "32GB", "1024M", "512MB", "1024K"]:
            spec = ResourceSpec(memory=memory)
            assert spec.memory == memory.upper()
    
    def test_invalid_memory_format(self):
        """Should reject invalid memory format."""
        from inferbench.core.models import ResourceSpec
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            ResourceSpec(memory="invalid")
    
    def test_valid_time_formats(self):
        """Should accept valid time formats."""
        from inferbench.core.models import ResourceSpec
        
        for time in ["01:00:00", "00:30:00", "10:00"]:
            spec = ResourceSpec(time=time)
            assert spec.time == time
    
    def test_invalid_time_format(self):
        """Should reject invalid time format."""
        from inferbench.core.models import ResourceSpec
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            ResourceSpec(time="invalid")
