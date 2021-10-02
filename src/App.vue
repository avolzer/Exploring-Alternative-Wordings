<template>
  <div id="app">
    <div class="form-group">
    <h2>Exploring alternative wordings</h2>
      <textarea 
        id = "userenglish" 
        name='text'
        v-model="inputText"
        rows="4" cols="50"
        required>
      </textarea><br><br>
      <button @click="getResult(inputText)">continue</button>
  </div>

   <div class="results"><br>
    <p v-for="alt in inputData.alternatives" class="tooltip">
      <button @click="getResult(inputText)" style="font-size: 20px;" class = "plain">
        {{ alt }}
      </button><br>
    </p>
    </div>
 </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      inputText: '',
      inputData: {"alternatives" : alternatives,
                  "scores" : scores
                          }
    };
  },

  methods: {
    async getResult(inputText){
        var url = new URL("/api/result", window.location);

        var params = {
          english:inputText,
        };
        url.searchParams.append('q', JSON.stringify(params));

        const res = await fetch(url);
        const input = await res.json();
        this.inputData = input;      
   },
  },
};
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
.yellowClass {
  background-color: yellow;
}
.results {
  text-align: left;
  margin-left: 20%;
  margin-right: 20%
}
.tooltip {
  position: relative;
  display: inline-block;
}
.tooltip .tooltiptext {
  visibility: hidden;
  width: 150px;
  background-color: gray;
  color: #fff;
  text-align: left;
  padding: 5px 0;
  border-radius: 6px;
  font-size: 66.67%;

  position: absolute;
  z-index: 1;
}
.tooltip:hover .tooltiptext { visibility: visible;}

button.plain { background:none; border:none; text-align: left;}

button.plain:hover { cursor: pointer;}

.grid-container {
  display: grid;
  grid-template-columns: auto auto auto;
  padding: 10px;
}
.grid-item {
  border: 1px solid rgba(0, 0, 0, 0.8);
  padding: 20px;
  font-size: 15px;
  text-align: left;
}
</style>
