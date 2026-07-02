import { createApp } from 'vue';
import ElementPlus from 'element-plus';
import App from './App.vue';
import router from './routers';
import store from './stores';
import 'element-plus/dist/index.css';
import './styles/index.css';

const app = createApp(App);

app.use(store);
app.use(router);
app.use(ElementPlus);
app.mount('#app');
