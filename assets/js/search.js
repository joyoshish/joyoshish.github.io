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
  
  // Function to open search modal
  function openSearchModal() {
    searchModal.style.display = 'block';
    setTimeout(() => {
      searchInput.focus();
    }, 100);
    document.body.classList.add('search-active');
  }
  
  // Function to close search modal
  function closeSearchModal() {
    searchModal.style.display = 'none';
    searchInput.value = '';
    searchResultsContainer.innerHTML = '';
    document.body.classList.remove('search-active');
  }
  
  // Keyboard shortcuts
  document.addEventListener('keydown', function(e) {
    // Ctrl+K or / to open search
    if ((e.ctrlKey && e.key === 'k') || (e.key === '/' && !['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName))) {
      e.preventDefault();
      openSearchModal();
    }
    
    // Escape to close search
    if (e.key === 'Escape' && searchModal.style.display === 'block') {
      closeSearchModal();
    }
  });
  
  // Click events
  searchButton.addEventListener('click', openSearchModal);
  searchClose.addEventListener('click', closeSearchModal);
  
  // Click outside to close
  window.addEventListener('click', function(e) {
    if (e.target === searchModal) {
      closeSearchModal();
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