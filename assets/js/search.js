// Search functionality
document.addEventListener('DOMContentLoaded', function() {
  // Variables
  const searchButton = document.getElementById('search-button');
  const searchModal = document.getElementById('search-modal');
  const searchInput = document.getElementById('search-input');
  const searchClose = document.getElementById('search-close');
  const searchResults = document.getElementById('search-results');
  const searchResultsContainer = searchResults.querySelector('.search-results-container');
  
  let searchIndex = [];
  let searchData = [];
  
  // Variables to track selected result
  let selectedIndex = -1;
  let searchResultItems = [];
  
  // Function to open search modal
  function openSearchModal() {
    searchModal.style.display = 'block';
    setTimeout(() => {
      searchInput.focus();
    }, 100);
    document.body.classList.add('search-active');
    resetSelection();
  }

  // Mobile-friendly search handling
  function adjustSearchForMobile() {
    if (window.innerWidth <= 768) {
      // On mobile, make sure search modal takes full screen height
      searchModal.style.paddingTop = "0";
      searchModal.style.paddingBottom = "0";
      
      // Prevent scrolling of body behind modal
      document.body.style.overflow = "hidden";
      
      // Ensure input doesn't zoom on iOS
      searchInput.setAttribute("autocomplete", "off");
      searchInput.setAttribute("autocorrect", "off");
      searchInput.setAttribute("autocapitalize", "off");
      searchInput.setAttribute("spellcheck", "false");
    } else {
      // Reset for desktop
      searchModal.style.paddingTop = "";
      searchModal.style.paddingBottom = "";
    }
  }
  
  // Function to close search modal
  function closeSearchModal() {
    searchModal.style.display = 'none';
    searchInput.value = '';
    searchResultsContainer.innerHTML = '';
    document.body.classList.remove('search-active');
    document.body.style.overflow = ""; // Reset body overflow
    resetSelection();
  }
  
  // Function to handle keyboard navigation
  function handleKeyboardNavigation(e) {
    if (!searchModal.style.display || searchModal.style.display === 'none') {
      return; // Don't do anything if search modal is not visible
    }
    
    searchResultItems = Array.from(searchResultsContainer.querySelectorAll('.search-result-item'));
    
    if (searchResultItems.length === 0) {
      return; // No results to navigate
    }
    
    // Handle arrow up/down navigation
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      selectedIndex = Math.min(selectedIndex + 1, searchResultItems.length - 1);
      updateSelectedResult();
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      selectedIndex = Math.max(selectedIndex - 1, 0);
      updateSelectedResult();
    } else if (e.key === 'Enter' && selectedIndex >= 0) {
      e.preventDefault();
      searchResultItems[selectedIndex].click(); // Navigate to the selected result
    }
  }
  
  // Function to update the selected result styling
  function updateSelectedResult() {
    // Remove selected class from all items
    searchResultItems.forEach((item) => {
      item.classList.remove('search-result-selected');
    });
    
    // Add selected class to current item
    if (selectedIndex >= 0 && selectedIndex < searchResultItems.length) {
      searchResultItems[selectedIndex].classList.add('search-result-selected');
      // Ensure the selected item is visible
      searchResultItems[selectedIndex].scrollIntoView({
        behavior: 'smooth',
        block: 'nearest'
      });
    }
  }
  
  // Function to reset selection
  function resetSelection() {
    selectedIndex = -1;
    searchResultItems = Array.from(searchResultsContainer.querySelectorAll('.search-result-item'));
    searchResultItems.forEach(item => {
      item.classList.remove('search-result-selected');
    });
  }
  
  // Keyboard shortcuts
  document.addEventListener('keydown', function(e) {
    // Ctrl+K or / to open search
    if ((e.ctrlKey && e.key === 'k') || (e.key === '/' && !['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName))) {
      e.preventDefault();
      openSearchModal();
      adjustSearchForMobile();
    }
    
    // Escape to close search
    if (e.key === 'Escape' && searchModal.style.display === 'block') {
      closeSearchModal();
    }
    
    // Handle arrow key navigation
    handleKeyboardNavigation(e);
  });
  
  // Click events
  searchButton.addEventListener('click', function() {
    openSearchModal();
    adjustSearchForMobile();
  });
  
  searchClose.addEventListener('click', closeSearchModal);
  
  // Click outside to close
  window.addEventListener('click', function(e) {
    if (e.target === searchModal) {
      closeSearchModal();
    }
  });
  
  // Also handle window resize
  window.addEventListener('resize', function() {
    if (searchModal.style.display === 'block') {
      adjustSearchForMobile();
    }
  });
  
  // Load search index
  function loadSearchIndex() {
    fetch('/search_index.json')
      .then(response => response.json())
      .then(data => {
        searchData = data;
        searchIndex = lunr(function() {
          this.field('title', { boost: 10 });
          this.field('categories', { boost: 5 });
          this.field('tags', { boost: 5 });
          this.field('content');
          this.ref('id');
          
          data.forEach(function(doc, idx) {
            this.add({
              'id': idx,
              'title': doc.title,
              'categories': doc.categories,
              'tags': doc.tags,
              'content': doc.content
            });
          }, this);
        });
      })
      .catch(error => console.error('Error loading search index:', error));
  }
  
  // Perform search
  searchInput.addEventListener('input', function() {
    const query = this.value.trim();
    resetSelection(); // Reset selection when input changes
    
    if (!query) {
      searchResultsContainer.innerHTML = '';
      return;
    }
    
    if (searchIndex.length === 0) {
      searchResultsContainer.innerHTML = '<div class="search-result-item">Loading search index...</div>';
      return;
    }
    
    const results = searchIndex.search(query);
    
    if (results.length === 0) {
      searchResultsContainer.innerHTML = '<div class="search-result-item">No results found</div>';
      return;
    }
    
    searchResultsContainer.innerHTML = '';
    results.slice(0, 10).forEach(result => {
      const item = searchData[parseInt(result.ref)];
      const resultElement = document.createElement('a');
      resultElement.href = item.url;
      resultElement.classList.add('search-result-item');
      
      const title = document.createElement('div');
      title.classList.add('search-result-title');
      title.textContent = item.title;
      
      const preview = document.createElement('div');
      preview.classList.add('search-result-preview');
      preview.textContent = item.content.substring(0, 150) + '...';
      
      resultElement.appendChild(title);
      resultElement.appendChild(preview);
      searchResultsContainer.appendChild(resultElement);
    });
  });
  
  // Load search index on page load
  loadSearchIndex();
});